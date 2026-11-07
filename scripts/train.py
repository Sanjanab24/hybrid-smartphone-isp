"""Fine-tune / train / distill the restoration networks.

The exact commands used for the reported results are listed in README.md
("Reproduce the results"). Short version:

  # teacher: pretrained NAFNet-SIDD (+ zero-initialised noise-map input), fine-tuned in fp32
  python -m scripts.train --model teacher --width 64 --pretrained weights/NAFNet-SIDD-width64.pth       --data data_shards/train --val data_shards/val --use-sidd-srgb --amp none --batch 1 --lr 2e-5       --iters 1500 --freeze-body-iters 300 --val-every 100 --out runs/teacher_fp32

  # student: our 0.44 M network, distilled from the teacher
  python -m scripts.train --model student --teacher runs/teacher_fp32/best.pt --distill 1.0       --data data_shards/train --val data_shards/val --use-sidd-srgb --amp bf16 --batch 8 --lr 5e-4       --iters 6000 --val-every 500 --out runs/student_v2

  # smoke test without SIDD (synthetic unprocessed images)
  python -m scripts.train --model student --synthetic eval_images --iters 50 --out runs/smoke
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.datasets import SiddShards, SyntheticUnprocessed
from isp.pipeline import RenderConfig
from models.losses import RestorationLoss
from models.nafnet import count_params, run_padded
from models.zoo import build, load_ckpt, save_ckpt


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["teacher", "student", "raw_denoiser", "e2e"])
    p.add_argument("--task", default=None, choices=["render_first", "raw_first", "e2e"])
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--pretrained", default=None, help="official NAFNet checkpoint for init")
    p.add_argument("--resume", default=None)
    p.add_argument("--data", default=None, help="SIDD shard dir (train)")
    p.add_argument("--val", default=None, help="SIDD shard dir (val)")
    p.add_argument("--synthetic", default=None, help="folder of clean images for unprocessed synthetic pairs")
    p.add_argument("--synthetic-mix", type=float, default=0.0, help="fraction of synthetic samples mixed into SIDD")
    p.add_argument("--synthetic-val", default=None, help="clean images for a second (synthetic) validation set")
    p.add_argument("--teacher", default=None, help="teacher checkpoint for distillation")
    p.add_argument("--distill", type=float, default=0.0, help="weight of L1(student, teacher)")
    p.add_argument("--no-noise-map", action="store_true", help="ablation: zero the noise-map channels")
    p.add_argument("--use-sidd-srgb", action="store_true", help="use SIDD's own noisy sRGB as input")
    p.add_argument("--est-map-prob", type=float, default=0.3)
    p.add_argument("--demosaic", default="mhc")
    p.add_argument("--crop", type=int, default=256)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--iters", type=int, default=20000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--w-ssim", type=float, default=0.2)
    p.add_argument("--w-color", type=float, default=0.002)
    p.add_argument("--w-lpips", type=float, default=0.0)
    p.add_argument("--freeze-body-iters", type=int, default=0, help="train only intro/ending first")
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--val-n", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "none"])
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def default_task(model):
    return {"teacher": "render_first", "student": "render_first", "raw_denoiser": "raw_first", "e2e": "e2e"}[model]


class Mixed(torch.utils.data.Dataset):
    def __init__(self, a, b, frac_b):
        self.a, self.b, self.frac_b = a, b, frac_b

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        if self.b is not None and np.random.rand() < self.frac_b:
            return self.b[i % len(self.b)]
        return self.a[i]


def make_loaders(args, task):
    cfg = RenderConfig(demosaic=args.demosaic)
    length = args.iters * args.batch
    syn = None
    if args.synthetic:
        syn = SyntheticUnprocessed(args.synthetic, task, args.crop, cfg, length=length, est_map_prob=args.est_map_prob)
    if args.data:
        tr = SiddShards(args.data, task, args.crop, cfg, args.est_map_prob, length=length,
                        use_sidd_srgb=args.use_sidd_srgb)
        train = Mixed(tr, syn, args.synthetic_mix) if syn is not None and args.synthetic_mix > 0 else tr
    elif syn is not None:
        train = syn
    else:
        raise SystemExit("need --data and/or --synthetic")
    if args.val:
        val = SiddShards(args.val, task, args.crop, cfg, 0.0, train=False, use_sidd_srgb=args.use_sidd_srgb)
        val.length = min(args.val_n, len(val.index))
    else:
        val = SyntheticUnprocessed(args.synthetic, task, args.crop, cfg, length=min(args.val_n, 32), train=False)
    kw = dict(num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    vkw = {**kw, "persistent_workers": False}
    vals = {"val": DataLoader(val, batch_size=1, shuffle=False, **vkw)}
    if args.synthetic_val:
        # second validation set in the app's domain (our rendering + simulated noise); the best
        # checkpoint is chosen on the MEAN of both, so neither domain is sacrificed
        sv = SyntheticUnprocessed(args.synthetic_val, task, args.crop, cfg, length=args.val_n, train=False)
        vals["syn_val"] = DataLoader(sv, batch_size=1, shuffle=False, **vkw)
    return DataLoader(train, batch_size=args.batch, shuffle=False, drop_last=True, **kw), vals


def validate_all(model, loaders, device, no_nmap, task):
    res = {k: validate(model, dl, device, no_nmap, task) for k, dl in loaders.items()}
    return float(np.mean(list(res.values()))), res


@torch.no_grad()
def validate(model, loader, device, no_nmap, task):
    model.eval()
    ps = []
    for inp, tgt in loader:
        inp, tgt = inp.to(device), tgt.to(device)
        if no_nmap and task == "render_first":
            inp[:, 3:] = 0
        out = run_padded(model, inp).clamp(0, 1)
        mse = torch.mean((out - tgt) ** 2).item()
        ps.append(10 * math.log10(1 / max(mse, 1e-12)))
    model.train()
    return float(np.mean(ps))


def main():
    args = parse()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    task = args.task or default_task(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=1)

    if args.resume:
        model = load_ckpt(args.resume, device).train()
        width = torch.load(args.resume, map_location="cpu", weights_only=False)["width"]
    else:
        model = build(args.model, args.width, args.pretrained).to(device).train()
        width = args.width or {"teacher": 64, "student": 16}.get(args.model, 32)
    teacher = None
    if args.teacher and args.distill > 0:
        teacher = load_ckpt(args.teacher, device)
        for p in teacher.parameters():
            p.requires_grad_(False)
        print(f"distilling from {args.teacher} ({count_params(teacher) / 1e6:.1f} M params)")

    train_dl, val_dl = make_loaders(args, task)
    crit = RestorationLoss(args.w_ssim, args.w_color if task != "raw_first" else 0.0, args.w_lpips).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.9))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.iters, pct_start=0.05,
                                                anneal_strategy="cos", final_div_factor=100)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp)
    if amp_dtype == torch.bfloat16 and device == "cuda" and not torch.cuda.is_bf16_supported():
        amp_dtype = torch.float16  # e.g. Kaggle T4/P100
    scaler = torch.amp.GradScaler(enabled=amp_dtype == torch.float16)
    body = [p for n, p in model.named_parameters() if not n.startswith(("intro", "ending"))]

    base, parts = validate_all(model, val_dl, device, args.no_noise_map, task)
    detail = lambda d: " ".join(f"{k} {v:.3f}" for k, v in d.items()) if len(d) > 1 else ""  # noqa: E731
    print(f"step 0 val PSNR {base:.3f} dB {detail(parts)}")
    best, log = base, [{"step": 0, "val_psnr": base, **parts}]
    save_ckpt(model, os.path.join(args.out, "best.pt"), args.model, width, {"val_psnr": best, "task": task})
    t0 = time.time()
    step = 0
    bad_steps = skipped = 0
    for inp, tgt in train_dl:
        step += 1
        if step > args.iters:
            break
        freeze = step <= args.freeze_body_iters
        for p in body:
            p.requires_grad_(not freeze)
        inp, tgt = inp.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
        if args.no_noise_map and task == "render_first":
            inp[:, 3:] = 0
        # only the networks run in reduced precision; the losses (SSIM, Lab) always run in fp32
        with torch.autocast(device_type=device, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            out = run_padded(model, inp)
            if teacher is not None:
                with torch.no_grad():
                    t_out = run_padded(teacher, inp)
        with torch.autocast(device_type=device, enabled=False):
            out = out.float()
            loss, logs = crit(out, tgt)
            if teacher is not None:
                kd = torch.nn.functional.l1_loss(out, t_out.float())
                loss = loss + args.distill * kd
                logs["kd"] = kd.item()
        sched.step()
        if not torch.isfinite(loss):  # never let one bad batch poison the weights
            bad_steps += 1
            opt.zero_grad(set_to_none=True)
            if bad_steps >= 50:
                print("stopping: 50 non-finite steps in a row; best.pt holds the last good model")
                break
            continue
        bad_steps = 0
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        if not torch.isfinite(gnorm):
            skipped += 1
            opt.zero_grad(set_to_none=True)
            scaler.update()
            continue
        scaler.step(opt)
        scaler.update()
        if step % 100 == 0:
            el = time.time() - t0
            print(f"step {step}/{args.iters} loss {loss.item():.4f} {json.dumps({k: round(v, 4) for k, v in logs.items()})} "
                  f"lr {sched.get_last_lr()[0]:.2e} {el / step:.3f}s/it", flush=True)
        if step % args.val_every == 0 or step == args.iters:
            v, parts = validate_all(model, val_dl, device, args.no_noise_map, task)
            log.append({"step": step, "val_psnr": v, **parts, "skipped_steps": skipped, **logs})
            print(f"step {step} val PSNR {v:.3f} dB (best {best:.3f}) {detail(parts)}", flush=True)
            if math.isfinite(v) and v > best:
                best = v
                save_ckpt(model, os.path.join(args.out, "best.pt"), args.model, width, {"val_psnr": best, "task": task})
            save_ckpt(model, os.path.join(args.out, "last.pt"), args.model, width, {"val_psnr": v, "task": task})
            with open(os.path.join(args.out, "log.json"), "w") as f:
                json.dump(log, f, indent=1)
    print(f"done. best val PSNR {best:.3f} dB (start {base:.3f})")


if __name__ == "__main__":
    main()
