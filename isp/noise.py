"""Noise modelling: heteroscedastic noise level function (NLF) and per-pixel noise maps.

Sensor noise is modelled as  var(x) = a * x + b  (shot + read noise) in
normalised raw units. The map is propagated through WB, CCM and gamma so the
restoration network sees the noise std of the image it actually receives.
"""
from __future__ import annotations

import cv2
import numpy as np

from .color import gamma_derivative, rgb_to_luma


def raw_noise_std(packed: np.ndarray, a: float, b: float, smooth: int = 5) -> np.ndarray:
    """Per-pixel noise std for a packed (H/2, W/2, 4) noisy raw image."""
    mu = cv2.blur(packed.astype(np.float32), (smooth, smooth), borderType=cv2.BORDER_REFLECT)
    return np.sqrt(np.maximum(a * np.clip(mu, 0, 1) + b, 1e-10)).astype(np.float32)


def estimate_nlf(packed: np.ndarray, n_bins: int = 16) -> tuple[float, float]:
    """Estimate (a, b) from a single noisy packed raw image.

    Uses the difference between the two green channels (same colour, almost the
    same scene content), whose variance is 2*var(noise), binned by intensity.
    """
    g1 = packed[..., 1].astype(np.float64)
    g2 = packed[..., 2].astype(np.float64)
    d = (g1 - g2) / np.sqrt(2.0)
    mu = 0.5 * (g1 + g2)
    # suppress content: only keep locally flat regions
    grad = np.abs(cv2.Laplacian(mu.astype(np.float32), cv2.CV_32F))
    flat = grad < np.percentile(grad, 50)
    mu_f, d_f = mu[flat], d[flat]
    lo, hi = np.percentile(mu_f, [1, 99])
    edges = np.linspace(lo, hi, n_bins + 1)
    xs, vs = [], []
    for i in range(n_bins):
        sel = (mu_f >= edges[i]) & (mu_f < edges[i + 1])
        if sel.sum() < 200:
            continue
        dd = d_f[sel]
        # robust variance via MAD
        sigma = 1.4826 * np.median(np.abs(dd - np.median(dd)))
        xs.append(mu_f[sel].mean())
        vs.append(sigma**2)
    if len(xs) < 2:
        v = float(np.var(d_f))
        return 0.0, max(v, 1e-8)
    A = np.stack([np.array(xs), np.ones(len(xs))], 1)
    sol, *_ = np.linalg.lstsq(A, np.array(vs), rcond=None)
    a, b = float(max(sol[0], 0.0)), float(max(sol[1], 1e-9))
    return a, b


def srgb_noise_map(
    packed_noisy: np.ndarray,
    a: float,
    b: float,
    wb_gains: np.ndarray,
    ccm: np.ndarray,
    lin_srgb_smooth: np.ndarray,
    gamma: str = "pow2.2",
    exposure_gain: float = 1.0,
) -> np.ndarray:
    """Propagate raw noise std to the rendered sRGB image -> (H, W, 3) std map.

    lin_srgb_smooth: smoothed linear sRGB image (after WB+demosaic+CCM, before
    gamma) at full resolution, used for the gamma derivative.
    """
    std = raw_noise_std(packed_noisy, a, b)
    std_rgb = np.stack([std[..., 0], np.sqrt(0.5 * (std[..., 1] ** 2 + std[..., 2] ** 2)), std[..., 3]], -1)
    std_rgb = std_rgb * wb_gains[None, None, :] * exposure_gain
    # independent channels through the linear CCM: var_out = sum_k M_ck^2 var_k
    var_out = np.einsum("hwc,kc->hwk", std_rgb**2, (ccm.astype(np.float32)) ** 2)
    std_lin = np.sqrt(np.maximum(var_out, 0))
    h, w = lin_srgb_smooth.shape[:2]
    std_lin = cv2.resize(std_lin, (w, h), interpolation=cv2.INTER_LINEAR)
    deriv = np.minimum(gamma_derivative(lin_srgb_smooth, gamma), 25.0)
    return np.clip(std_lin * deriv, 0, 0.5).astype(np.float32)


def estimate_srgb_noise_map(srgb: np.ndarray, n_bins: int = 12) -> np.ndarray:
    """Noise map for an sRGB image when no raw/NLF is available (e.g. SIDD sRGB blocks).

    Fits std(intensity) from high-pass residuals in flat regions and evaluates it
    per pixel on a smoothed image. Training mixes this estimate with the exact
    propagated map so the network tolerates either.
    """
    x = srgb.astype(np.float32)
    smooth = cv2.GaussianBlur(x, (0, 0), 1.5)
    resid = x - cv2.blur(x, (3, 3))
    y = rgb_to_luma(smooth)
    grad = np.abs(cv2.Laplacian(y, cv2.CV_32F))
    flat = grad < np.percentile(grad, 40)
    out = np.zeros_like(x)
    edges = np.linspace(0, 1, n_bins + 1)
    for c in range(3):
        mu = smooth[..., c][flat]
        r = resid[..., c][flat]
        xs, ss = [], []
        for i in range(n_bins):
            sel = (mu >= edges[i]) & (mu < edges[i + 1])
            if sel.sum() < 50:
                continue
            rr = r[sel]
            # residual of a 3x3 box removes 1/9 of the noise: scale by sqrt(9/8)
            ss.append(1.4826 * np.median(np.abs(rr - np.median(rr))) * np.sqrt(9 / 8))
            xs.append(mu[sel].mean())
        if not xs:
            out[..., c] = 1.4826 * np.median(np.abs(resid[..., c]))
            continue
        out[..., c] = np.interp(smooth[..., c], np.array(xs), np.array(ss))
    return np.clip(out, 0, 0.5).astype(np.float32)


def add_raw_noise(clean_bayer: np.ndarray, a: float, b: float, rng: np.random.Generator) -> np.ndarray:
    """Synthesise heteroscedastic Gaussian noise on a clean mosaic."""
    var = a * np.clip(clean_bayer, 0, 1) + b
    return (clean_bayer + rng.standard_normal(clean_bayer.shape).astype(np.float32) * np.sqrt(var)).astype(
        np.float32
    )


def sample_nlf(rng: np.random.Generator, level: float | None = None) -> tuple[float, float]:
    """Brooks et al. (CVPR 2019) shot/read noise prior. level in [0,1] picks a quantile."""
    lo, hi = np.log(1e-4), np.log(1.2e-2)
    u = rng.uniform() if level is None else float(np.clip(level, 0, 1))
    log_shot = lo + u * (hi - lo)
    log_read = 2.18 * log_shot + 1.20 + (rng.normal(0, 0.26) if level is None else 0.0)
    return float(np.exp(log_shot)), float(np.exp(log_read))
