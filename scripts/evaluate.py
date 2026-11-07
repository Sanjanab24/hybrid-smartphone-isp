"""Comparative evaluation: traditional vs learned vs hybrid ISP, plus ablations.

Protocols
  sidd_raw   : full RAW->sRGB on held-out SIDD crops (data_shards/test = unseen camera)
  sidd_srgb  : SIDD validation sRGB blocks (1280 x 256^2), the standard benchmark for the restoration stage
  synthetic  : any clean images -> simulated phone RAW at several noise levels (unseen content)

  python -m scripts.evaluate --protocol sidd_raw --data data_shards/test \
      --teacher runs/teacher/best.pt --student runs/student/best.pt --student-int8 weights/student_int8.onnx \
      --e2e runs/e2e/best.pt --raw-denoiser runs/raw/best.pt --pretrained weights/NAFNet-SIDD-width64.pth
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
from tqdm import tqdm

from isp.bayer import pack_rggb, unpack_rggb
from isp.hybrid import HybridISP
from isp.metrics import LPIPS, all_metrics
from isp.noise import estimate_srgb_noise_map, raw_noise_std
from isp.pipeline import RenderConfig, render, traditional_isp
from isp.raw_io import RawImage, from_srgb, load_mat_array
from isp.router import ClassicalRestorer, OnnxRestorer
from models.nafnet import run_padded
from models.zoo import TorchRestorer, build, load_ckpt

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ data
def iter_sidd_raw(root, limit=0):
    dirs = sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.exists(os.path.join(d, "meta.json")))
    n = 0
    for d in dirs:
        meta = json.load(open(os.path.join(d, "meta.json")))
        noisy = np.load(os.path.join(d, "noisy_raw.npy"), mmap_mode="r")
        gt = np.load(os.path.join(d, "gt_srgb.npy"), mmap_mode="r")
        ns_path = os.path.join(d, "noisy_srgb.npy")
        ns = np.load(ns_path, mmap_mode="r") if os.path.exists(ns_path) else None
        for k in range(noisy.shape[0]):
            m = dict(meta)
            if ns is not None:
                m["noisy_srgb"] = np.asarray(ns[k], np.float32) / 255
            raw = RawImage(np.asarray(noisy[k], np.float32), np.array(meta["wb_gains"], np.float32),
                           np.array(meta["cam2srgb"], np.float32), tuple(meta["nlf"]), "sidd",
                           gt_srgb=np.asarray(gt[k], np.float32) / 255, meta=m)
            yield f"{meta['key']}#{k}", meta.get("camera", "?"), raw
            n += 1
            if limit and n >= limit:
                return


def iter_synthetic(folder, levels, limit=0):
    files = sorted(f for e in ("png", "jpg", "jpeg") for f in glob.glob(os.path.join(folder, f"*.{e}")))
    n = 0
    for f in files:
        img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        for lv in levels:
            yield f"{os.path.basename(f)}@{lv}", f"level_{lv}", from_srgb(img, lv, seed=n)
            n += 1
            if limit and n >= limit:
                return


def iter_sidd_srgb(val_dir, limit=0):
    noisy = load_mat_array(os.path.join(val_dir, "ValidationNoisyBlocksSrgb.mat"))
    gt = load_mat_array(os.path.join(val_dir, "ValidationGtBlocksSrgb.mat"))
    n = 0
    for i in range(noisy.shape[0]):
        for j in range(noisy.shape[1]):
            yield f"{i}_{j}", "sidd_val", noisy[i, j].astype(np.float32) / 255, gt[i, j].astype(np.float32) / 255
            n += 1
            if limit and n >= limit:
                return


# ------------------------------------------------------------------ variants
def build_variants(args):
    """Return {name: fn(raw) -> sRGB} for RAW protocols and {name: restorer} for the sRGB protocol."""
    cfg = RenderConfig(demosaic=args.demosaic)
    teacher = TorchRestorer(load_ckpt(args.teacher, DEV), "teacher", DEV, half=args.half) if args.teacher else None
    # ablation: a teacher fine-tuned identically but with the noise map zeroed in training and test
    teacher_nonm = (TorchRestorer(load_ckpt(args.teacher_nonm, DEV), "teacher", DEV, zero_noise_map=True)
                    if args.teacher_nonm else None)
    student_nokd = TorchRestorer(load_ckpt(args.student_nokd, DEV), "student", DEV) if args.student_nokd else None
    pretrained = None
    if args.pretrained:
        pretrained = TorchRestorer(build("teacher", args.pretrained_width, args.pretrained, verbose=False), "pretrained",
                                   DEV, zero_noise_map=True)
    student = TorchRestorer(load_ckpt(args.student, DEV), "student", DEV) if args.student else None
    s_int8 = OnnxRestorer(args.student_int8, "student") if args.student_int8 else None
    e2e = load_ckpt(args.e2e, DEV) if args.e2e else None
    rawd = load_ckpt(args.raw_denoiser, DEV) if args.raw_denoiser else None
    classical = ClassicalRestorer()

    srgb_variants = {"noisy_input": None, "A_classical_guided": classical}
    if pretrained:
        srgb_variants["D0_pretrained_nafnet_zero_shot"] = pretrained
    if teacher:
        srgb_variants["D_teacher_finetuned"] = teacher
    if teacher_nonm:
        srgb_variants["abl_teacher_no_noise_map"] = teacher_nonm
    if student:
        srgb_variants["E_student_fp32"] = student
    if student_nokd:
        srgb_variants["abl_student_no_distillation"] = student_nokd
    if s_int8:
        srgb_variants["E_student_int8"] = s_int8

    raw_variants = {}
    raw_variants["noisy_render"] = lambda raw: render(raw, cfg).srgb
    raw_variants["A_traditional_isp"] = lambda raw: traditional_isp(raw, cfg)[0]

    if e2e is not None:
        @torch.no_grad()
        def f_e2e(raw):
            x = torch.from_numpy(pack_rggb(raw.bayer).transpose(2, 0, 1)[None].copy()).to(DEV)
            return run_padded(e2e, x)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        raw_variants["B_learned_isp_e2e"] = f_e2e

    if rawd is not None:
        @torch.no_grad()
        def f_rawfirst(raw):
            pk = pack_rggb(raw.bayer)
            x = np.concatenate([pk, raw_noise_std(pk, *raw.nlf)], -1).transpose(2, 0, 1)[None]
            y = run_padded(rawd, torch.from_numpy(x.copy()).to(DEV))[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            clean = RawImage(unpack_rggb(y), raw.wb_gains, raw.cam2srgb, raw.nlf, raw.source)
            return render(clean, cfg).srgb
        raw_variants["C_hybrid_raw_first"] = f_rawfirst

    use_sidd = getattr(args, "use_sidd_srgb", False)

    def hybrid(restorers, budget, random_routing=False):
        h = HybridISP(restorers, cfg, tile=args.tile, overlap=args.overlap)
        return lambda raw: h.process(raw, budget, random_routing=random_routing,
                                     srgb_override=raw.meta.get("noisy_srgb") if use_sidd else None)["output"]

    if use_sidd:  # sRGB-domain variants all start from SIDD's own rendering (as in training)
        raw_variants["noisy_input_sidd_srgb"] = lambda raw: raw.meta["noisy_srgb"]
    raw_variants["A2_classical_guided_tiles"] = hybrid({}, "classical_only")
    if pretrained:
        raw_variants["D0_render_then_pretrained_nafnet"] = hybrid({"teacher": pretrained}, "teacher_only")
    if teacher:
        raw_variants["D_hybrid_render_first_teacher"] = hybrid({"teacher": teacher}, "teacher_only")
    if teacher_nonm:
        raw_variants["abl_D_no_noise_map"] = hybrid({"teacher": teacher_nonm}, "teacher_only")
    stu = s_int8 or student
    if student:
        raw_variants["E_hybrid_student_fp32"] = hybrid({"student": student}, "student_only")
    if student_nokd:
        raw_variants["abl_E_student_no_distillation"] = hybrid({"student": student_nokd}, "student_only")
    if s_int8:
        raw_variants["E_hybrid_student_int8"] = hybrid({"student": s_int8}, "student_only")
    if stu and teacher:
        for b in ("low", "balanced", "high"):
            raw_variants[f"F_router_{b}"] = hybrid({"student": stu, "teacher": teacher}, b)
        raw_variants["abl_F_random_routing_balanced"] = hybrid({"student": stu, "teacher": teacher}, "balanced", True)
    return raw_variants, srgb_variants


def run(args):
    raw_variants, srgb_variants = build_variants(args)
    lp = LPIPS(DEV) if not args.no_lpips else None
    agg = defaultdict(lambda: defaultdict(list))
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    times = defaultdict(list)

    if args.protocol == "sidd_srgb":
        for key, grp, noisy, gt in tqdm(iter_sidd_srgb(args.sidd_val_dir, args.limit), desc="sidd_srgb"):
            nmap = estimate_srgb_noise_map(noisy)
            x = np.concatenate([noisy, nmap], -1).transpose(2, 0, 1)[None].astype(np.float32)
            for name, fn in srgb_variants.items():
                t0 = time.perf_counter()
                out = noisy if fn is None else fn(x)[0].transpose(1, 2, 0)
                times[name].append(time.perf_counter() - t0)
                for k, v in all_metrics(out, gt, lp).items():
                    agg[name][k].append(v)
    else:
        it = (iter_sidd_raw(args.data, args.limit) if args.protocol == "sidd_raw"
              else iter_synthetic(args.images, args.levels, args.limit))
        os.makedirs(args.save_examples, exist_ok=True) if args.save_examples else None
        for idx, (key, grp, raw) in enumerate(tqdm(it, desc=args.protocol)):
            gt = raw.gt_srgb
            for name, fn in raw_variants.items():
                if DEV == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = fn(raw)
                if DEV == "cuda":
                    torch.cuda.synchronize()
                times[name].append(time.perf_counter() - t0)
                h, w = min(out.shape[0], gt.shape[0]), min(out.shape[1], gt.shape[1])
                m = all_metrics(out[8:h - 8, 8:w - 8], gt[8:h - 8, 8:w - 8], lp)
                for k, v in m.items():
                    agg[name][k].append(v)
                    groups[grp][name][k].append(v)
                if args.save_examples and idx < args.n_examples:
                    cv2.imwrite(os.path.join(args.save_examples, f"{idx:03d}_{name}.png"),
                                cv2.cvtColor((np.clip(out, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            if args.save_examples and idx < args.n_examples:
                cv2.imwrite(os.path.join(args.save_examples, f"{idx:03d}_gt.png"),
                            cv2.cvtColor((gt * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    summary = {name: {**{k: float(np.mean(v)) for k, v in d.items()}, "n": len(next(iter(d.values()))),
                      "latency_s": float(np.median(times[name]))} for name, d in agg.items()}
    by_group = {g: {n: {k: float(np.mean(v)) for k, v in d.items()} for n, d in gd.items()} for g, gd in groups.items()}
    out = {"protocol": args.protocol, "device": DEV, "torch_gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else None,
           "summary": summary, "by_group": by_group}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n{'variant':40s} {'PSNR':>7s} {'SSIM':>6s} {'dE00':>6s} {'LPIPS':>6s} {'lat(s)':>7s}")
    for name, s in summary.items():
        print(f"{name:40s} {s['psnr']:7.2f} {s['ssim']:6.4f} {s['delta_e']:6.3f} {s.get('lpips', float('nan')):6.4f} "
              f"{s['latency_s']:7.3f}")
    print(f"wrote {args.out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", required=True, choices=["sidd_raw", "sidd_srgb", "synthetic"])
    p.add_argument("--data", default="data_shards/test")
    p.add_argument("--sidd-val-dir", default="datasets/sidd_val")
    p.add_argument("--images", default="eval_images")
    p.add_argument("--levels", type=float, nargs="+", default=[0.2, 0.5, 0.8])
    p.add_argument("--teacher"), p.add_argument("--student"), p.add_argument("--student-int8")
    p.add_argument("--e2e"), p.add_argument("--raw-denoiser")
    p.add_argument("--teacher-nonm", help="ablation teacher trained with --no-noise-map")
    p.add_argument("--student-nokd", help="ablation student trained without distillation")
    p.add_argument("--pretrained", help="official NAFNet-SIDD checkpoint (zero-shot baseline)")
    p.add_argument("--pretrained-width", type=int, default=64)
    p.add_argument("--demosaic", default="mhc")
    p.add_argument("--tile", type=int, default=256)
    p.add_argument("--overlap", type=int, default=32)
    p.add_argument("--half", action="store_true")
    p.add_argument("--use-sidd-srgb", action="store_true",
                   help="sidd_raw: sRGB-domain variants take SIDD's own noisy sRGB (how the models were trained)")
    p.add_argument("--no-lpips", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--save-examples", default=None)
    p.add_argument("--n-examples", type=int, default=6)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    args.out = args.out or f"results/eval_{args.protocol}.json"
    run(args)


if __name__ == "__main__":
    main()
