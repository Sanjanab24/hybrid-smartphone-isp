"""Image-quality metrics: PSNR, SSIM, CIEDE2000 Delta E (numpy) and LPIPS (torch, optional)."""
from __future__ import annotations

import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(x: np.ndarray, ref: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(np.clip(ref, 0, 1), np.clip(x, 0, 1), data_range=1.0))


def ssim(x: np.ndarray, ref: np.ndarray) -> float:
    return float(structural_similarity(np.clip(ref, 0, 1), np.clip(x, 0, 1), channel_axis=-1, data_range=1.0))


def delta_e2000(x: np.ndarray, ref: np.ndarray) -> float:
    """Mean CIEDE2000 between two sRGB images (D65)."""
    return float(np.mean(deltaE_ciede2000(rgb2lab(np.clip(ref, 0, 1)), rgb2lab(np.clip(x, 0, 1)))))


class LPIPS:
    """Lazy LPIPS (AlexNet) wrapper; only used in offline evaluation."""

    def __init__(self, device: str | None = None):
        import lpips
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = lpips.LPIPS(net="alex", verbose=False).to(self.device).eval()

    def __call__(self, x: np.ndarray, ref: np.ndarray) -> float:
        t = self.torch
        to = lambda a: t.from_numpy(np.clip(a, 0, 1).transpose(2, 0, 1)[None].copy()).float().to(self.device) * 2 - 1  # noqa: E731
        with t.no_grad():
            return float(self.net(to(x), to(ref)).item())


def all_metrics(x: np.ndarray, ref: np.ndarray, lpips_fn: LPIPS | None = None) -> dict:
    m = {"psnr": psnr(x, ref), "ssim": ssim(x, ref), "delta_e": delta_e2000(x, ref)}
    if lpips_fn is not None:
        m["lpips"] = lpips_fn(x, ref)
    return m
