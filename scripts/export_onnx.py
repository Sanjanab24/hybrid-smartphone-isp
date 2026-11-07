"""Export a trained restoration model to ONNX (dynamic H, W) after a stability gate, and verify it.

The gate runs the model on random images, extreme images (black, white, flat grey,
saturated colour, checkerboard) and noise maps from 0 to 1. Export is refused if any
output is non-finite or far outside [0, 1], because the app must never return garbage
for an unusual upload.

  python -m scripts.export_onnx --ckpt runs/student/best.pt --out weights/student_fp32.onnx
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from models.nafnet import run_padded
from models.zoo import load_ckpt


def stress_inputs(cin: int, size: int = 256):
    g = torch.Generator().manual_seed(0)
    imgs = {
        "random": torch.rand(3, size, size, generator=g),
        "black": torch.zeros(3, size, size),
        "white": torch.ones(3, size, size),
        "grey": torch.full((3, size, size), 0.5),
        "saturated_red": torch.stack([torch.ones(size, size), torch.zeros(size, size), torch.zeros(size, size)]),
        "checkerboard": ((torch.arange(size)[:, None] // 8 + torch.arange(size)[None] // 8) % 2).float().expand(3, -1, -1),
    }
    for name, img in imgs.items():
        for nm in (0.0, 0.02, 0.1, 0.3, 1.0):
            x = torch.cat([img, torch.full((cin - 3, size, size), nm)], 0) if cin > 3 else img
            yield f"{name}/noise={nm}", x[None]


def stability_gate(model, cin: int) -> None:
    fails = []
    worst = 0.0
    with torch.no_grad():
        for name, x in stress_inputs(cin):
            y = run_padded(model, x)
            if not torch.isfinite(y).all():
                fails.append(f"{name}: non-finite output")
                continue
            m = float(y.abs().max())
            worst = max(worst, m)
            if m > 4.0:
                fails.append(f"{name}: output magnitude {m:.3g}")
    if fails:
        raise SystemExit("STABILITY GATE FAILED - not exporting:\n  " + "\n  ".join(fails[:10]))
    print(f"stability gate passed (30 stress inputs, max |output| {worst:.2f})")


def export(ckpt: str, out: str, opset: int = 17, size: int = 256):
    model = load_ckpt(ckpt, "cpu").float()
    if model.out_shuffle == 1 and model.in_ch == 6:
        stability_gate(model, model.in_ch)
    x = torch.rand(1, model.in_ch, size, size)
    if model.in_ch == 6:
        x[:, 3:] *= 0.1  # realistic noise-map range for the trace/verification input
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    kw = dict(input_names=["input"], output_names=["output"], opset_version=opset,
              dynamic_axes={"input": {0: "n", 2: "h", 3: "w"}, "output": {0: "n", 2: "h2", 3: "w2"}})
    try:
        torch.onnx.export(model, (x,), out, dynamo=False, **kw)
    except TypeError:  # older torch without the dynamo flag
        torch.onnx.export(model, (x,), out, **kw)
    import onnxruntime as ort

    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    for s in (size, 128 + 64):
        xi = torch.rand(2, model.in_ch, s, s)
        if model.in_ch == 6:
            xi[:, 3:] *= 0.1
        ref = model(xi).detach().numpy()
        got = sess.run(None, {"input": xi.numpy()})[0]
        err = float(np.abs(ref - got).max())
        print(f"verify {s}x{s}: max |torch - onnx| = {err:.2e}")
        if not err < 1e-3:
            raise SystemExit("ONNX output does not match PyTorch")
    print(f"exported {out} ({os.path.getsize(out) / 1e6:.2f} MB)")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--opset", type=int, default=17)
    a = p.parse_args()
    export(a.ckpt, a.out, a.opset)


if __name__ == "__main__":
    main()
