"""Day-0 check: does our classical front-end reproduce SIDD's own sRGB rendering?

For a few images per camera, renders SIDD noisy RAW with every combination of
CFA pattern, colour matrix, row normalisation, demosaic and gamma, compares
against SIDD's noisy sRGB, and writes the best settings per camera to
configs/sidd_render.json. A PSNR above ~40 dB means the pretrained sRGB
network receives in-domain inputs from our pipeline.

  python -m scripts.check_rendering --raw-root <SIDD_Medium_Raw> --srgb-root <SIDD_Medium_Srgb> --per-camera 2
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from collections import defaultdict

import cv2
import numpy as np

from data.sidd import index_sidd
from isp.bayer import PATTERNS, to_rggb
from isp.color import cam2srgb_from_xyz2cam, wb_from_as_shot_neutral
from isp.metrics import psnr
from isp.pipeline import RenderConfig, render
from isp.raw_io import RawImage, load_mat_array, read_sidd_metadata


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", required=True)
    p.add_argument("--srgb-root", required=True)
    p.add_argument("--per-camera", type=int, default=2)
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--out", default="configs/sidd_render.json")
    args = p.parse_args()

    items = [i for i in index_sidd(args.raw_root, args.srgb_root) if i.noisy_srgb]
    by_cam = defaultdict(list)
    for it in items:
        if len(by_cam[it.camera]) < args.per_camera:
            by_cam[it.camera].append(it)
    grid = list(itertools.product(PATTERNS, ["color_matrix1", "color_matrix2"], [True, False],
                                  ["mhc", "ea", "bilinear"], ["pow2.2", "srgb"]))
    result = {}
    for cam, its in sorted(by_cam.items()):
        scores = defaultdict(list)
        for it in its:
            bayer_full = np.clip(load_mat_array(it.noisy_raw).astype(np.float32), 0, 1)
            meta = read_sidd_metadata(it.meta)
            ref_full = cv2.cvtColor(cv2.imread(it.noisy_srgb), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            if ref_full.shape[:2] != bayer_full.shape:
                print(f"{it.key}: sRGB {ref_full.shape[:2]} vs RAW {bayer_full.shape} - orientation differs")
            H, W = bayer_full.shape
            y0, x0 = (H - args.crop) // 4 * 2, (W - args.crop) // 4 * 2
            wb = wb_from_as_shot_neutral(meta["as_shot_neutral"])
            for pat, cm, rn, dm, gm in grid:
                if meta[cm] is None:
                    continue
                b, (dy, dx) = to_rggb(bayer_full[y0:y0 + args.crop + 2, x0:x0 + args.crop + 2], pat)
                ref = ref_full[y0 + dy:y0 + dy + b.shape[0], x0 + dx:x0 + dx + b.shape[1]]
                raw = RawImage(b, wb, cam2srgb_from_xyz2cam(meta[cm], rownorm_xyz2cam=rn), (1e-4, 1e-6), "sidd")
                out = render(raw, RenderConfig(demosaic=dm, gamma=gm)).srgb
                scores[(pat, cm, rn, dm, gm)].append(psnr(out[4:-4, 4:-4], ref[4:-4, 4:-4]))
            print(f"{cam} {it.key}: meta CFA={meta['cfa_pattern']} NoiseProfile={meta['noise_profile']}")
        ranked = sorted(((np.mean(v), k) for k, v in scores.items()), reverse=True)
        best_psnr, (pat, cm, rn, dm, gm) = ranked[0]
        print(f"{cam}: best {best_psnr:.2f} dB with pattern={pat} {cm} rownorm={rn} demosaic={dm} gamma={gm}")
        for s, k in ranked[1:4]:
            print(f"      {s:.2f} dB {k}")
        result[cam] = {"pattern": pat, "color_matrix": cm, "rownorm": rn,
                       "render": {"demosaic": dm, "gamma": gm}, "match_psnr": round(float(best_psnr), 2)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"wrote {args.out}")
    worst = min(v["match_psnr"] for v in result.values()) if result else 0
    print("PASS: rendering matches SIDD" if worst > 40 else
          "NOTE: rendering differs from SIDD's (fine-tuning absorbs this; or train with --use-sidd-srgb)")


if __name__ == "__main__":
    main()
