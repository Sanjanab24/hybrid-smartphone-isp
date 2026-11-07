"""Model factory: teacher, student, RAW-domain denoiser and end-to-end learned ISP.

| name          | input                         | output       | init                         |
|---------------|-------------------------------|--------------|------------------------------|
| teacher       | sRGB(3) + noise map(3)        | sRGB(3)      | NAFNet-SIDD (pretrained)     |
| student       | sRGB(3) + noise map(3)        | sRGB(3)      | scratch, distilled           |
| raw_denoiser  | packed RAW(4) + noise std(4)  | packed RAW(4)| NAFNet-SIDD body             |
| e2e           | packed RAW(4)                 | sRGB(3) @2x  | NAFNet-SIDD body             |
"""
from __future__ import annotations

import os

import torch

from .nafnet import NAFNet, count_params

SIDD_BLOCKS = dict(middle_blk_num=12, enc_blk_nums=(2, 2, 4, 8), dec_blk_nums=(2, 2, 2, 2))

# Our student: compact 3-level NAFNet, width 16, about 0.44 M parameters.
# LayerNorm is kept: a norm-free variant was faster but its activations blew up
# (to 1e21) on out-of-distribution inputs, which is unacceptable for unseen photos.
STUDENT_CFG = dict(width=16, middle_blk_num=2, enc_blk_nums=(1, 1, 2), dec_blk_nums=(1, 1, 1), use_norm=True)


def _load_state(path: str) -> dict:
    sd = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("params_ema", "params", "state_dict", "model"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
            break
    return {k.removeprefix("module."): v for k, v in sd.items()}


def load_pretrained_body(model: NAFNet, path: str, expand_intro: bool = True) -> dict:
    """Load an official 3-channel NAFNet checkpoint into a (possibly wider-input) model.

    Image channels of `intro` get the pretrained weights; extra input channels
    (noise map) are zero-initialised, so at step 0 the model equals the
    pretrained network exactly. Shape-mismatched layers are skipped and reported.
    """
    sd = _load_state(path)
    own = model.state_dict()
    loaded, skipped = [], []
    for k, v in sd.items():
        if k not in own:
            skipped.append(k)
            continue
        if own[k].shape == v.shape:
            own[k] = v
            loaded.append(k)
        elif k == "intro.weight" and expand_intro and own[k].shape[0] == v.shape[0] and own[k].shape[1] > v.shape[1]:
            w = torch.zeros_like(own[k])
            w[:, : v.shape[1]] = v
            own[k] = w
            loaded.append(k)
        else:
            skipped.append(k)
    model.load_state_dict(own)
    return {"loaded": len(loaded), "skipped": skipped}


def build(name: str, width: int | None = None, pretrained: str | None = None, verbose: bool = True) -> NAFNet:
    if name == "teacher":
        m = NAFNet(in_ch=6, out_ch=3, width=width or 64, **SIDD_BLOCKS)
    elif name == "student":
        cfg = dict(STUDENT_CFG)
        if width:
            cfg["width"] = width
        m = NAFNet(in_ch=6, out_ch=3, **cfg)
    elif name == "raw_denoiser":
        m = NAFNet(in_ch=8, out_ch=4, width=width or 32, **SIDD_BLOCKS)
    elif name == "e2e":
        m = NAFNet(in_ch=4, out_ch=3, width=width or 32, residual=False, out_shuffle=2, **SIDD_BLOCKS)
    else:
        raise ValueError(name)
    if pretrained:
        if not os.path.exists(pretrained):
            raise FileNotFoundError(f"pretrained weights not found: {pretrained} (see README: Download weights)")
        info = load_pretrained_body(m, pretrained)
        if verbose:
            print(f"[{name}] loaded {info['loaded']} tensors from {pretrained}; skipped {len(info['skipped'])}: "
                  f"{info['skipped'][:6]}")
    if verbose:
        print(f"[{name}] params: {count_params(m) / 1e6:.3f} M")
    return m


def save_ckpt(model: NAFNet, path: str, name: str, width: int, extra: dict | None = None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"name": name, "width": width, "state_dict": model.state_dict(), **(extra or {})}, path)


def load_ckpt(path: str, device: str = "cpu") -> NAFNet:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = build(ck["name"], width=ck["width"], verbose=False)
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval()


class TorchRestorer:
    """Wrap a sRGB-domain model as a router path: (N,6,H,W) numpy -> (N,3,H,W) numpy."""

    def __init__(self, model: NAFNet, name: str, device: str = "cuda", half: bool = False, zero_noise_map: bool = False):
        self.model = model.to(device).eval()
        self.device, self.half, self.name, self.zero = device, half and device == "cuda", name, zero_noise_map
        if self.half:
            self.model.half()

    @torch.no_grad()
    def __call__(self, x):
        from .nafnet import run_padded

        t = torch.from_numpy(x).to(self.device)
        if self.zero:
            t[:, 3:] = 0
        t = t.half() if self.half else t.float()
        return run_padded(self.model, t).float().clamp(0, 1).cpu().numpy()
