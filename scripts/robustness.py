"""Robustness study (advanced requirement 5): degrade inputs in controlled ways and measure each variant.

Conditions, applied to clean RAW (SIDD GT RAW from the test shards, or unprocessed clean images):
  noise_x{k}      noise variance scaled by k (higher ISO than seen in training)
  under_{e}       exposure scaled by e (then digital gain 1/e in the pipeline)
  wb_err          as-shot white balance off by +20% red / -20% blue (bad metadata)
  wb_err+grayworld  same, recovered by the gray-world fallback

  python -m scripts.robustness --data data_shards/test --teacher runs/teacher/best.pt --student-int8 weights/student_int8.onnx
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import cv2
import numpy as np

from isp.bayer import pack_rggb
from isp.color import wb_gray_world
from isp.metrics import all_metrics
from isp.noise import add_raw_noise
from isp.pipeline import RenderConfig, render
from isp.raw_io import RawImage
from isp.unprocess import unprocess
from scripts.evaluate import build_variants

KEEP = ["A_traditional_isp", "C_hybrid_raw_first", "D_hybrid_render_first_teacher", "E_hybrid_student_int8",
        "E_hybrid_student_fp32", "F_router_balanced"]


def clean_sources(args):
    """Yield (clean_bayer, meta-like dict, gt_srgb)."""
    cfg = RenderConfig(demosaic=args.demosaic)
    if args.data and os.path.isdir(args.data):
        dirs = sorted(d for d in glob.glob(os.path.join(args.data, "*")) if os.path.exists(os.path.join(d, "gt_raw.npy")))
        for d in dirs[: args.limit]:
            meta = json.load(open(os.path.join(d, "meta.json")))
            gt_raw = np.load(os.path.join(d, "gt_raw.npy"), mmap_mode="r")
            c = min(args.crop, gt_raw.shape[1])
            b = np.asarray(gt_raw[0, :c, :c], np.float32)
            m = {"wb": np.array(meta["wb_gains"], np.float32), "ccm": np.array(meta["cam2srgb"], np.float32),
                 "nlf": tuple(meta["nlf"]), "render": meta.get("render", {})}
            gt = render(RawImage(b, m["wb"], m["ccm"], m["nlf"], meta=m), cfg).srgb  # clean render = reference
            yield b, m, gt
    else:
        files = sorted(glob.glob(os.path.join(args.images, "*.jpg")) + glob.glob(os.path.join(args.images, "*.png")))
        for i, f in enumerate(files[: args.limit]):
            img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            u = unprocess(img[: args.crop, : args.crop], rng=np.random.default_rng(i), noise_level=0.4)
            yield u["bayer_clean"], {"wb": u["wb_gains"], "ccm": u["cam2srgb"], "nlf": u["nlf"]}, u["clean_srgb"]


def conditions(clean, m, rng):
    a, b = m["nlf"]
    for k in (1, 2, 4, 8):
        yield f"noise_x{k}", RawImage(np.clip(add_raw_noise(clean, k * a, k * b, rng), 0, 1), m["wb"], m["ccm"],
                                      (k * a, k * b), meta={"render": m.get("render", {})})
    for e in (0.5, 0.25):
        noisy = np.clip(add_raw_noise(clean * e, a, b, rng), 0, 1)
        yield f"under_x{e}", RawImage(noisy, m["wb"], m["ccm"], (a, b), exposure_gain=1 / e,
                                      meta={"render": m.get("render", {})})
    noisy = np.clip(add_raw_noise(clean, 2 * a, 2 * b, rng), 0, 1)
    bad = m["wb"] * np.array([1.2, 1.0, 0.8], np.float32)
    yield "wb_err", RawImage(noisy, bad, m["ccm"], (2 * a, 2 * b), meta={"render": m.get("render", {})})
    yield "wb_err+grayworld", RawImage(noisy, wb_gray_world(pack_rggb(noisy)), m["ccm"], (2 * a, 2 * b),
                                       meta={"render": m.get("render", {})})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data_shards/test")
    p.add_argument("--images", default="eval_images")
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--teacher"), p.add_argument("--student"), p.add_argument("--student-int8")
    p.add_argument("--e2e"), p.add_argument("--raw-denoiser"), p.add_argument("--pretrained")
    p.add_argument("--teacher-nonm"), p.add_argument("--student-nokd")
    p.add_argument("--pretrained-width", type=int, default=64)
    p.add_argument("--demosaic", default="mhc")
    p.add_argument("--tile", type=int, default=256)
    p.add_argument("--overlap", type=int, default=32)
    p.add_argument("--half", action="store_true")
    p.add_argument("--out", default="results/robustness.json")
    a = p.parse_args()
    raw_variants, _ = build_variants(a)
    variants = {k: v for k, v in raw_variants.items() if k in KEEP}
    rng = np.random.default_rng(0)
    res = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for clean, m, gt in clean_sources(a):
        for cond, raw in conditions(clean, m, rng):
            for name, fn in variants.items():
                out = fn(raw)
                for k, v in all_metrics(out[8:-8, 8:-8], gt[8:-8, 8:-8]).items():
                    res[cond][name][k].append(v)
    summary = {c: {n: {k: float(np.mean(v)) for k, v in d.items()} for n, d in cd.items()} for c, cd in res.items()}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(summary, f, indent=1)
    names = list(variants)
    print(f"{'condition':18s} " + " ".join(f"{n[:22]:>22s}" for n in names))
    for c, cd in summary.items():
        print(f"{c:18s} " + " ".join(f"{cd[n]['psnr']:9.2f}dB {cd[n]['delta_e']:5.2f}dE" for n in names))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        conds = [c for c in summary if c.startswith("noise_x")]
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        for n in names:
            ax[0].plot([int(c[7:]) for c in conds], [summary[c][n]["psnr"] for c in conds], marker="o", label=n)
            ax[1].plot([int(c[7:]) for c in conds], [summary[c][n]["delta_e"] for c in conds], marker="o", label=n)
        for x, t in zip(ax, ("PSNR (dB) vs noise scale", "Delta E 2000 vs noise scale")):
            x.set_xscale("log", base=2)
            x.set_title(t)
            x.grid(alpha=0.3)
        ax[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(a.out.replace(".json", ".png"), dpi=130)
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
