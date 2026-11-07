"""Classical denoising and sharpening (edge-preserving)."""
from __future__ import annotations

import cv2
import numpy as np

from .color import rgb_to_luma


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """He et al. guided filter (gray guide, per-channel src), O(N) with box filters."""
    k = (2 * radius + 1, 2 * radius + 1)
    box = lambda x: cv2.boxFilter(x, -1, k, borderType=cv2.BORDER_REFLECT)  # noqa: E731
    I = guide.astype(np.float32)
    mean_I = box(I)
    var_I = box(I * I) - mean_I**2
    out = np.empty_like(src, dtype=np.float32)
    chans = src[..., None] if src.ndim == 2 else src
    for c in range(chans.shape[-1]):
        p = chans[..., c].astype(np.float32)
        mean_p = box(p)
        cov = box(I * p) - mean_I * mean_p
        a = cov / (var_I + eps)
        b = mean_p - a * mean_I
        res = box(a) * I + box(b)
        if src.ndim == 2:
            return res
        out[..., c] = res
    return out


def guided_denoise(srgb: np.ndarray, sigma: float) -> np.ndarray:
    """Self-guided filter; eps tied to the noise variance (adaptive strength)."""
    y = rgb_to_luma(srgb)
    eps = max((1.5 * sigma) ** 2, 1e-6)
    return np.clip(guided_filter(y, srgb, radius=2, eps=eps), 0, 1)


def bilateral_denoise(srgb: np.ndarray, sigma: float) -> np.ndarray:
    s = float(np.clip(sigma * 3.0, 0.01, 0.5))
    return np.clip(cv2.bilateralFilter(srgb.astype(np.float32), d=7, sigmaColor=s, sigmaSpace=2.0), 0, 1)


def nlm_denoise(srgb: np.ndarray, sigma: float) -> np.ndarray:
    """OpenCV fast non-local means; h follows the measured noise std."""
    u8 = (np.clip(srgb, 0, 1) * 255 + 0.5).astype(np.uint8)
    h = float(np.clip(sigma * 255 * 0.9, 1, 40))
    out = cv2.fastNlMeansDenoisingColored(u8, None, h, h, 5, 15)
    return out.astype(np.float32) / 255


def adaptive_unsharp(srgb: np.ndarray, noise_std: float, base_amount: float = 0.6,
                     sigma_ref: float = 0.03, radius: float = 1.0) -> np.ndarray:
    """Unsharp mask on luma; strength falls to zero as residual noise rises."""
    amount = base_amount * float(np.clip(1.0 - noise_std / sigma_ref, 0.0, 1.0))
    if amount <= 1e-3:
        return srgb
    y = rgb_to_luma(srgb)
    detail = y - cv2.GaussianBlur(y, (0, 0), radius)
    return np.clip(srgb + amount * detail[..., None], 0, 1).astype(np.float32)
