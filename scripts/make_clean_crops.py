"""Cut clean training crops from SIDD ground-truth sRGB images, for synthetic RAW training.

Only the images already in data_shards/train and data_shards/val are used, so the
held-out test camera never leaks into training. The crops are later unprocessed
into simulated noisy RAW (isp/unprocess.py) and rendered by OUR pipeline, which
teaches the networks the noise texture the app produces for JPG/PNG/DNG inputs.

  python -m scripts.make_clean_crops --srgb-root "C:/Datasets/SIDD/hf/SRGB/train" --shards data_shards --out data_shards/clean
"""
from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np


def shard_keys(split_dir: str) -> list[tuple[str, str]]:
    """data_shards/<split>/<folder>_<suffix> -> (folder, suffix)."""
    out = []
    for d in sorted(glob.glob(os.path.join(split_dir, "*"))):
        if os.path.exists(os.path.join(d, "meta.json")):
            name = os.path.basename(d)
            folder, suffix = name.rsplit("_", 1)
            out.append((folder, suffix))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--srgb-root", required=True)
    p.add_argument("--shards", default="data_shards")
    p.add_argument("--out", default="data_shards/clean")
    p.add_argument("--crops", type=int, default=40, help="crops per training image")
    p.add_argument("--val-crops", type=int, default=6)
    p.add_argument("--size", type=int, default=512)
    a = p.parse_args()

    gt_files = {}
    for f in glob.glob(os.path.join(a.srgb_root, "**", "*_GT_SRGB_*.PNG"), recursive=True):
        folder = os.path.basename(os.path.dirname(f))
        suffix = os.path.basename(f).rsplit("_", 1)[1].split(".")[0]
        gt_files[(folder, suffix)] = f
    rng = np.random.default_rng(0)
    for split, n in (("train", a.crops), ("val", a.val_crops)):
        keys = shard_keys(os.path.join(a.shards, split))
        out = os.path.join(a.out, split)
        os.makedirs(out, exist_ok=True)
        written = 0
        for folder, suffix in keys:
            f = gt_files.get((folder, suffix))
            if f is None:
                print(f"no GT sRGB for {folder}/{suffix}, skipped")
                continue
            img = cv2.imread(f, cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            for i in range(n):
                y = int(rng.integers(0, h - a.size))
                x = int(rng.integers(0, w - a.size))
                cv2.imwrite(os.path.join(out, f"{folder}_{suffix}_{i:02d}.png"), img[y:y + a.size, x:x + a.size])
                written += 1
        print(f"{split}: {written} clean crops from {len(keys)} images -> {out}")


if __name__ == "__main__":
    main()
