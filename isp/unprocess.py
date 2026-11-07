"""Invert the rendering of an sRGB image into a plausible noisy Bayer RAW.

Based on Brooks et al., "Unprocessing Images for Learned Raw Denoising" (CVPR 2019).
The inverse uses exactly the stages of our forward pipeline (gamma, CCM, WB,
mosaic), so a noiseless round trip reproduces the input up to demosaic error.
That gives ground truth for JPEG/PNG inputs in the demo and for synthetic data.
"""
from __future__ import annotations

import numpy as np

from .bayer import mosaic_rggb
from .color import SRGB2XYZ, gamma_decode
from .noise import add_raw_noise, sample_nlf

# XYZ->camera matrices of four cameras (from Brooks et al.)
_XYZ2CAMS = np.array(
    [
        [[1.0234, -0.2969, -0.2266], [-0.5625, 1.6328, -0.0469], [-0.0703, 0.2188, 0.6406]],
        [[0.4913, -0.0541, -0.0202], [-0.613, 1.3513, 0.2906], [-0.1564, 0.2151, 0.7183]],
        [[0.838, -0.263, -0.0639], [-0.2887, 1.0725, 0.2496], [-0.0627, 0.1427, 0.5438]],
        [[0.6596, -0.2079, -0.0562], [-0.4782, 1.3016, 0.1933], [-0.097, 0.1581, 0.5181]],
    ]
)


def random_cam2srgb(rng: np.random.Generator | None) -> np.ndarray:
    if rng is None:
        w = np.full(4, 0.25)
    else:
        w = rng.uniform(0, 1, size=4)
        w = w / w.sum()
    xyz2cam = np.tensordot(w, _XYZ2CAMS, axes=1)
    rgb2cam = xyz2cam @ SRGB2XYZ
    rgb2cam = rgb2cam / rgb2cam.sum(axis=1, keepdims=True)
    return np.linalg.inv(rgb2cam).astype(np.float32)


def random_wb(rng: np.random.Generator | None) -> np.ndarray:
    if rng is None:
        return np.array([2.15, 1.0, 1.7], np.float32)
    return np.array([rng.uniform(1.9, 2.4), 1.0, rng.uniform(1.5, 1.9)], np.float32)


def unprocess(
    srgb: np.ndarray,
    rng: np.random.Generator | None = None,
    gamma: str = "pow2.2",
    noise_level: float | None = None,
    cam2srgb: np.ndarray | None = None,
    wb_gains: np.ndarray | None = None,
):
    """sRGB (H, W, 3) in [0,1] -> dict(bayer_noisy, bayer_clean, wb_gains, cam2srgb, nlf, clean_srgb).

    noise_level in [0, 1] chooses a quantile of the Brooks noise prior (0 = ISO ~100,
    1 = very high ISO); None samples it randomly.
    """
    gen = rng if rng is not None else np.random.default_rng(0)
    h, w = srgb.shape[:2]
    srgb = srgb[: h - h % 2, : w - w % 2].astype(np.float32)
    cam2srgb = random_cam2srgb(rng) if cam2srgb is None else cam2srgb
    wb_gains = random_wb(rng) if wb_gains is None else wb_gains
    lin = gamma_decode(srgb, gamma)
    cam = np.einsum("hwc,kc->hwk", lin, np.linalg.inv(cam2srgb).astype(np.float32))
    # inverse WB with highlight protection (Brooks' safe_invert_gains, simplified)
    inv = 1.0 / wb_gains
    gray = cam.mean(-1, keepdims=True)
    inflection = 0.9
    mask = (np.maximum(gray - inflection, 0) / (1 - inflection)) ** 2
    safe_inv = np.maximum(mask + (1 - mask) * inv[None, None, :], inv[None, None, :])
    cam = np.clip(cam * safe_inv, 0, 1)
    clean = mosaic_rggb(cam)
    a, b = sample_nlf(gen, noise_level)
    noisy = np.clip(add_raw_noise(clean, a, b, gen), 0, 1)
    return {
        "bayer_noisy": noisy.astype(np.float32),
        "bayer_clean": clean.astype(np.float32),
        "wb_gains": wb_gains.astype(np.float32),
        "cam2srgb": cam2srgb.astype(np.float32),
        "nlf": (a, b),
        "clean_srgb": srgb,
    }
