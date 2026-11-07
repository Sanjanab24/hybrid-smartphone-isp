"""Bayer CFA utilities.

Everything downstream assumes an RGGB mosaic. Other layouts are converted by
cropping one row and/or column (lossless, keeps native pixels).
"""
from __future__ import annotations

import numpy as np

PATTERNS = ("RGGB", "BGGR", "GRBG", "GBRG")

# (dy, dx) crop that turns a mosaic with this layout into RGGB
_RGGB_OFFSET = {"RGGB": (0, 0), "GRBG": (0, 1), "GBRG": (1, 0), "BGGR": (1, 1)}


def pattern_from_cfa(cfa: np.ndarray, color_desc: str = "RGBG") -> str:
    """Convert a 2x2 CFA index array (rawpy style) to a pattern string."""
    cfa = np.asarray(cfa)[:2, :2]
    letters = [color_desc[int(i)] for i in cfa.flatten()]
    pat = "".join(letters)
    if pat not in PATTERNS:
        raise ValueError(f"Unsupported CFA pattern {pat!r}")
    return pat


def to_rggb(bayer: np.ndarray, pattern: str) -> tuple[np.ndarray, tuple[int, int]]:
    """Crop a mosaic so it starts with an R pixel, with even height/width.

    Returns the cropped mosaic and the (dy, dx) offset applied, which is needed
    to crop aligned reference images (e.g. SIDD ground-truth sRGB) identically.
    """
    pattern = pattern.upper()
    if pattern not in _RGGB_OFFSET:
        raise ValueError(f"Unsupported CFA pattern {pattern!r}; expected one of {PATTERNS}")
    dy, dx = _RGGB_OFFSET[pattern]
    out = bayer[dy:, dx:]
    h, w = out.shape[:2]
    out = out[: h - h % 2, : w - w % 2]
    return out, (dy, dx)


def pack_rggb(bayer: np.ndarray) -> np.ndarray:
    """(H, W) RGGB mosaic -> (H/2, W/2, 4) with channels R, Gr, Gb, B."""
    return np.stack(
        [bayer[0::2, 0::2], bayer[0::2, 1::2], bayer[1::2, 0::2], bayer[1::2, 1::2]], axis=-1
    )


def unpack_rggb(packed: np.ndarray) -> np.ndarray:
    """(H/2, W/2, 4) -> (H, W) RGGB mosaic."""
    h, w, _ = packed.shape
    out = np.empty((h * 2, w * 2), dtype=packed.dtype)
    out[0::2, 0::2] = packed[..., 0]
    out[0::2, 1::2] = packed[..., 1]
    out[1::2, 0::2] = packed[..., 2]
    out[1::2, 1::2] = packed[..., 3]
    return out


def mosaic_rggb(rgb: np.ndarray) -> np.ndarray:
    """Sample an (H, W, 3) image with an RGGB CFA -> (H, W) mosaic."""
    h, w = rgb.shape[:2]
    rgb = rgb[: h - h % 2, : w - w % 2]
    out = np.empty(rgb.shape[:2], dtype=rgb.dtype)
    out[0::2, 0::2] = rgb[0::2, 0::2, 0]
    out[0::2, 1::2] = rgb[0::2, 1::2, 1]
    out[1::2, 0::2] = rgb[1::2, 0::2, 1]
    out[1::2, 1::2] = rgb[1::2, 1::2, 2]
    return out


def flip_rggb(bayer: np.ndarray, flip_x: bool, flip_y: bool) -> np.ndarray:
    """Flip an RGGB mosaic and re-crop so it is still RGGB (loses 2 px per flipped axis)."""
    out = bayer
    if flip_x:
        out = out[:, ::-1][:, 1:-1]
    if flip_y:
        out = out[::-1][1:-1]
    return np.ascontiguousarray(out)


def apply_wb_mosaic(bayer: np.ndarray, gains: np.ndarray) -> np.ndarray:
    """Multiply an RGGB mosaic by per-colour white-balance gains (r, g, b)."""
    out = bayer.astype(np.float32, copy=True)
    out[0::2, 0::2] *= gains[0]
    out[0::2, 1::2] *= gains[1]
    out[1::2, 0::2] *= gains[1]
    out[1::2, 1::2] *= gains[2]
    return out


def bayer_visualization(bayer: np.ndarray) -> np.ndarray:
    """False-colour view of the mosaic (each pixel shown in its CFA colour)."""
    h, w = bayer.shape
    vis = np.zeros((h, w, 3), np.float32)
    vis[0::2, 0::2, 0] = bayer[0::2, 0::2]
    vis[0::2, 1::2, 1] = bayer[0::2, 1::2]
    vis[1::2, 0::2, 1] = bayer[1::2, 0::2]
    vis[1::2, 1::2, 2] = bayer[1::2, 1::2]
    return np.clip(vis ** (1 / 2.2), 0, 1)
