"""Static INT8 post-training quantisation (ONNX Runtime, QDQ, per-channel weights).

Calibration uses real model inputs (rendered sRGB + noise map) from SIDD shards
or from synthetic unprocessed images, so activation ranges match deployment.

  python -m scripts.quantize --model weights/student_fp32.onnx --out weights/student_int8.onnx --calib data_shards/val
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process


class Reader(CalibrationDataReader):
    def __init__(self, samples):
        self.it = iter([{"input": s} for s in samples])

    def get_next(self):
        return next(self.it, None)


def calib_samples(src: str, n: int, crop: int, use_sidd_srgb: bool = False):
    from data.datasets import SiddShards, SyntheticUnprocessed

    if os.path.isdir(src) and any(os.path.exists(os.path.join(src, d, "meta.json")) for d in os.listdir(src)):
        ds = SiddShards(src, "render_first", crop, train=False, use_sidd_srgb=use_sidd_srgb)
        ds.length = min(n, len(ds.index))
    else:
        ds = SyntheticUnprocessed(src, "render_first", crop, length=n, train=False)
    return [ds[i][0].numpy()[None] for i in range(len(ds))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--calib", required=True, help="SIDD shard dir or folder of clean images")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--crop", type=int, default=256)
    p.add_argument("--method", default="percentile", choices=["minmax", "entropy", "percentile"])
    p.add_argument("--use-sidd-srgb", action="store_true", help="SIDD calibration inputs use SIDD's own sRGB")
    p.add_argument("--calib-synthetic", default=None,
                   help="also calibrate on simulated RAW from these clean images (the app's input domain)")
    a = p.parse_args()

    pre = a.out.replace(".onnx", "_pre.onnx")
    # symbolic shape inference cannot resolve the dynamic H/W of the U-Net; ONNX shape inference is enough
    quant_pre_process(a.model, pre, skip_symbolic_shape=True)
    samples = calib_samples(a.calib, a.n, a.crop, a.use_sidd_srgb)
    if a.calib_synthetic:
        samples += calib_samples(a.calib_synthetic, a.n, a.crop)
    print(f"calibrating on {len(samples)} inputs")
    method = {"minmax": CalibrationMethod.MinMax, "entropy": CalibrationMethod.Entropy,
              "percentile": CalibrationMethod.Percentile}[a.method]
    quantize_static(pre, a.out, Reader(samples), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, calibrate_method=method,
                    extra_options={"CalibPercentile": 99.99} if a.method == "percentile" else {})
    os.remove(pre)
    import onnxruntime as ort

    s32 = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"])
    s8 = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
    errs = []
    for x in samples[:8]:
        y32 = s32.run(None, {"input": x})[0]
        y8 = s8.run(None, {"input": x})[0]
        errs.append(10 * np.log10(1 / max(np.mean((np.clip(y32, 0, 1) - np.clip(y8, 0, 1)) ** 2), 1e-12)))
    print(f"INT8 vs FP32 output agreement: {np.mean(errs):.2f} dB PSNR")
    print(f"size: fp32 {os.path.getsize(a.model) / 1e6:.2f} MB -> int8 {os.path.getsize(a.out) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
