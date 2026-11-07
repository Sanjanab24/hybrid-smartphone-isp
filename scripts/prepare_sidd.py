"""Convert SIDD-Medium (RAW + sRGB) into memory-mappable crop shards with a leave-one-camera-out split.

  python -m scripts.prepare_sidd --raw-root /kaggle/input/sidd-medium-raw --srgb-root /kaggle/input/sidd-medium-srgb \
      --out data_shards --test-camera G4 --crops 8 --crop 512
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from tqdm import tqdm

from data.sidd import index_sidd, load_camera_cfg, split_items, write_item_crops


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", required=True)
    p.add_argument("--srgb-root", required=True)
    p.add_argument("--out", default="data_shards")
    p.add_argument("--test-camera", default="G4", help="camera held out entirely (unseen-sensor test)")
    p.add_argument("--crops", type=int, default=8, help="random crops per training image")
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--test-crops", type=int, default=4)
    p.add_argument("--test-crop", type=int, default=1024)
    p.add_argument("--camera-cfg", default="configs/sidd_render.json")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    items = index_sidd(args.raw_root, args.srgb_root)
    if args.limit:
        items = items[: args.limit]
    print(f"found {len(items)} SIDD pairs; cameras: {sorted({i.camera for i in items})}")
    missing = sum(i.gt_srgb is None for i in items)
    if missing:
        print(f"warning: {missing} items without GT sRGB (render_first/e2e tasks need it)")
    train, val, test = split_items(items, args.test_camera)
    cam_cfg = load_camera_cfg(args.camera_cfg)
    rng = np.random.default_rng(0)
    summary = {}
    for name, its, n, c in [("train", train, args.crops, args.crop), ("val", val, 2, args.crop),
                            ("test", test, args.test_crops, args.test_crop)]:
        out = os.path.join(args.out, name)
        os.makedirs(out, exist_ok=True)
        ok = 0
        for it in tqdm(its, desc=name):
            try:
                write_item_crops(it, out, n, c, cam_cfg, rng)
                ok += 1
            except Exception as e:  # keep going; report at the end
                print(f"skip {it.key}: {e}")
        summary[name] = {"images": ok, "cameras": sorted({i.camera for i in its})}
    with open(os.path.join(args.out, "split.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
