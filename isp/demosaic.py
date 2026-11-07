"""Demosaicing for RGGB mosaics: bilinear, Malvar-He-Cutler (MHC) and OpenCV EA."""
from __future__ import annotations

import cv2
import numpy as np

_BORDER = cv2.BORDER_REFLECT_101  # keeps CFA parity at image borders


def _masks(h: int, w: int):
    r = np.zeros((h, w), np.float32)
    g = np.zeros((h, w), np.float32)
    b = np.zeros((h, w), np.float32)
    r[0::2, 0::2] = 1
    g[0::2, 1::2] = 1
    g[1::2, 0::2] = 1
    b[1::2, 1::2] = 1
    return r, g, b


def _f(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    return cv2.filter2D(img, -1, k.astype(np.float32), borderType=_BORDER)


def demosaic_bilinear(bayer: np.ndarray) -> np.ndarray:
    bayer = bayer.astype(np.float32)
    mr, mg, mb = _masks(*bayer.shape)
    k_g = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], np.float32) / 4
    k_rb = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 4
    return np.stack([_f(bayer * mr, k_rb), _f(bayer * mg, k_g), _f(bayer * mb, k_rb)], -1)


# Malvar, He & Cutler (ICASSP 2004) gradient-corrected linear interpolation kernels (/8)
_K_G_AT_RB = np.array(
    [[0, 0, -1, 0, 0], [0, 0, 2, 0, 0], [-1, 2, 4, 2, -1], [0, 0, 2, 0, 0], [0, 0, -1, 0, 0]], np.float32
) / 8
_K_ROW = np.array(  # colour from horizontal neighbours (e.g. R at G in an R row)
    [[0, 0, 0.5, 0, 0], [0, -1, 0, -1, 0], [-1, 4, 5, 4, -1], [0, -1, 0, -1, 0], [0, 0, 0.5, 0, 0]], np.float32
) / 8
_K_COL = _K_ROW.T.copy()  # colour from vertical neighbours
_K_DIAG = np.array(  # R at B / B at R
    [[0, 0, -1.5, 0, 0], [0, 2, 0, 2, 0], [-1.5, 0, 6, 0, -1.5], [0, 2, 0, 2, 0], [0, 0, -1.5, 0, 0]], np.float32
) / 8


def demosaic_mhc(bayer: np.ndarray) -> np.ndarray:
    x = bayer.astype(np.float32)
    h, w = x.shape
    g_rb = _f(x, _K_G_AT_RB)
    row = _f(x, _K_ROW)
    col = _f(x, _K_COL)
    diag = _f(x, _K_DIAG)

    r = np.empty_like(x)
    g = np.empty_like(x)
    b = np.empty_like(x)
    # R sites (even, even)
    r[0::2, 0::2] = x[0::2, 0::2]
    g[0::2, 0::2] = g_rb[0::2, 0::2]
    b[0::2, 0::2] = diag[0::2, 0::2]
    # Gr sites (even row = R row, odd col)
    r[0::2, 1::2] = row[0::2, 1::2]
    g[0::2, 1::2] = x[0::2, 1::2]
    b[0::2, 1::2] = col[0::2, 1::2]
    # Gb sites (odd row = B row, even col)
    r[1::2, 0::2] = col[1::2, 0::2]
    g[1::2, 0::2] = x[1::2, 0::2]
    b[1::2, 0::2] = row[1::2, 0::2]
    # B sites (odd, odd)
    r[1::2, 1::2] = diag[1::2, 1::2]
    g[1::2, 1::2] = g_rb[1::2, 1::2]
    b[1::2, 1::2] = x[1::2, 1::2]
    return np.clip(np.stack([r, g, b], -1), 0, None)


def demosaic_opencv(bayer: np.ndarray, method: str = "EA") -> np.ndarray:
    """OpenCV edge-aware / VNG demosaic. OpenCV names RGGB sensors 'BayerBG'."""
    code = {"EA": cv2.COLOR_BayerBG2RGB_EA, "VNG": cv2.COLOR_BayerBG2RGB_VNG}[method]
    u16 = np.clip(bayer * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
    if method == "VNG":  # VNG only supports 8-bit
        u8 = (u16 >> 8).astype(np.uint8)
        return cv2.demosaicing(u8, code).astype(np.float32) / 255.0
    return cv2.demosaicing(u16, code).astype(np.float32) / 65535.0


def demosaic(bayer: np.ndarray, method: str = "mhc") -> np.ndarray:
    method = method.lower()
    if method == "bilinear":
        return demosaic_bilinear(bayer)
    if method == "mhc":
        return demosaic_mhc(bayer)
    if method in ("ea", "opencv_ea"):
        return demosaic_opencv(bayer, "EA")
    if method in ("vng", "opencv_vng"):
        return demosaic_opencv(bayer, "VNG")
    raise ValueError(f"unknown demosaic method {method!r}")
