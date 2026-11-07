"""White balance, colour correction, gamma, tone mapping and colour-space helpers."""
from __future__ import annotations

import numpy as np

XYZ2SRGB = np.array(
    [[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560], [0.0556434, -0.2040259, 1.0572252]],
    np.float64,
)
SRGB2XYZ = np.linalg.inv(XYZ2SRGB)


# ---------------------------------------------------------------- white balance
def wb_from_as_shot_neutral(asn) -> np.ndarray:
    """DNG AsShotNeutral (camera response to white) -> per-channel gains, green = 1."""
    asn = np.asarray(asn, np.float64).reshape(-1)[:3]
    gains = 1.0 / np.maximum(asn, 1e-6)
    return (gains / gains[1]).astype(np.float32)


def wb_gray_world(packed: np.ndarray, clip: float = 0.95) -> np.ndarray:
    """Gray-world estimate on a packed RGGB image, ignoring clipped/near-black pixels."""
    rgb = np.stack([packed[..., 0], 0.5 * (packed[..., 1] + packed[..., 2]), packed[..., 3]], -1)
    valid = np.all((rgb < clip) & (rgb > 0.01), axis=-1)
    if valid.sum() < 100:
        valid = np.ones(rgb.shape[:2], bool)
    means = rgb[valid].mean(0)
    gains = means[1] / np.maximum(means, 1e-6)
    return np.clip(gains, 0.25, 8.0).astype(np.float32)


def wb_white_patch(packed: np.ndarray, pct: float = 99.0) -> np.ndarray:
    rgb = np.stack([packed[..., 0], 0.5 * (packed[..., 1] + packed[..., 2]), packed[..., 3]], -1)
    lum = rgb.mean(-1)
    thr = np.percentile(lum, pct)
    sel = (lum >= thr) & np.all(rgb < 0.98, -1)
    if sel.sum() < 20:
        return wb_gray_world(packed)
    means = rgb[sel].mean(0)
    return np.clip(means[1] / np.maximum(means, 1e-6), 0.25, 8.0).astype(np.float32)


# ---------------------------------------------------------------- colour correction
def cam2srgb_from_xyz2cam(xyz2cam, rownorm_xyz2cam: bool = True, rownorm_out: bool = True) -> np.ndarray:
    """Build the camera->linear sRGB matrix from a DNG ColorMatrix (XYZ->camera).

    Follows the SIDD simple-camera-pipeline / Brooks et al. convention: rows of
    XYZ->cam are normalised to sum to 1 so that white maps to white after WB.
    """
    m = np.asarray(xyz2cam, np.float64).reshape(-1)[:9].reshape(3, 3)
    if rownorm_xyz2cam:
        m = m / np.sum(m, axis=1, keepdims=True)
    cam2xyz = np.linalg.inv(m)
    cam2srgb = XYZ2SRGB @ cam2xyz
    if rownorm_out:
        cam2srgb = cam2srgb / np.sum(cam2srgb, axis=1, keepdims=True)
    return cam2srgb.astype(np.float32)


def apply_ccm(rgb: np.ndarray, ccm: np.ndarray) -> np.ndarray:
    return np.einsum("hwc,kc->hwk", rgb, ccm.astype(np.float32)).astype(np.float32)


# ---------------------------------------------------------------- gamma / tone
def gamma_encode(lin: np.ndarray, mode: str = "pow2.2") -> np.ndarray:
    x = np.clip(lin, 0.0, 1.0)
    if mode == "pow2.2":
        return x ** (1 / 2.2)
    if mode == "srgb":
        return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    raise ValueError(mode)


def gamma_decode(srgb: np.ndarray, mode: str = "pow2.2") -> np.ndarray:
    x = np.clip(srgb, 0.0, 1.0)
    if mode == "pow2.2":
        return x ** 2.2
    if mode == "srgb":
        return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    raise ValueError(mode)


def gamma_derivative(lin: np.ndarray, mode: str = "pow2.2", floor: float = 1e-3) -> np.ndarray:
    """d gamma(x)/dx, used to propagate noise std through the tone curve."""
    x = np.clip(lin, floor, 1.0)
    if mode == "pow2.2":
        return (1 / 2.2) * x ** (1 / 2.2 - 1)
    return np.where(x <= 0.0031308, 12.92, 1.055 / 2.4 * x ** (1 / 2.4 - 1))


def smoothstep_tone(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return 3 * x**2 - 2 * x**3


def auto_exposure_gain(lin: np.ndarray, target: float = 0.85, pct: float = 99.0, max_gain: float = 8.0) -> float:
    """Digital gain for under-exposed captures (adaptive stage, off for SIDD eval)."""
    p = float(np.percentile(lin.mean(-1), pct))
    if p <= 1e-4:
        return max_gain
    return float(np.clip(target / p, 1.0, max_gain))


# ---------------------------------------------------------------- colour spaces
def rgb_to_luma(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
