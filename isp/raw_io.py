"""Input loading and validation: DNG (any phone), SIDD .MAT, and sRGB images.

Every loader returns a RawImage holding a normalised RGGB mosaic plus the
metadata the classical stages need. Problems the user can fix raise InputError
with a readable message (shown in the app instead of a stack trace).
"""
from __future__ import annotations

import csv
import io
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from .bayer import PATTERNS, pattern_from_cfa, to_rggb
from .color import cam2srgb_from_xyz2cam, wb_from_as_shot_neutral, wb_gray_world
from .noise import estimate_nlf
from .unprocess import unprocess

MAX_FILE_MB = 80
MIN_SIDE = 64


class InputError(ValueError):
    """An input problem that should be reported to the user, not crash the app."""


@dataclass
class RawImage:
    bayer: np.ndarray  # (H, W) float32 RGGB mosaic, black level removed, in [0, 1]
    wb_gains: np.ndarray  # (3,) r, g, b multipliers, g = 1
    cam2srgb: np.ndarray  # (3, 3) camera RGB -> linear sRGB
    nlf: tuple[float, float]  # noise variance = a * x + b in normalised raw units
    source: str = "unknown"
    exposure_gain: float = 1.0
    gt_srgb: np.ndarray | None = None  # reference image when one exists
    meta: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def shape(self):
        return self.bayer.shape

    def crop(self, y: int, x: int, h: int, w: int) -> "RawImage":
        """Crop on even coordinates (keeps RGGB phase). Reference image is cropped too."""
        y, x, h, w = y - y % 2, x - x % 2, h - h % 2, w - w % 2
        gt = None if self.gt_srgb is None else self.gt_srgb[y : y + h, x : x + w]
        return RawImage(
            self.bayer[y : y + h, x : x + w].copy(), self.wb_gains, self.cam2srgb, self.nlf, self.source,
            self.exposure_gain, gt, dict(self.meta), list(self.notes),
        )

    def center_crop_pixels(self, max_pixels: int) -> "RawImage":
        h, w = self.bayer.shape
        if h * w <= max_pixels:
            return self
        s = np.sqrt(max_pixels / (h * w))
        ch, cw = int(h * s), int(w * s)
        out = self.crop((h - ch) // 2, (w - cw) // 2, ch, cw)
        out.notes.append(f"Large input: processed the centre {cw}x{ch} crop (native pixels, no resampling).")
        return out


# ----------------------------------------------------------------------- .mat helpers
def load_mat_array(path: str, key: str | None = None) -> np.ndarray:
    """Load an array from MATLAB v5 (scipy) or v7.3 (HDF5) .mat files."""
    import scipy.io

    try:
        d = scipy.io.loadmat(path)
        keys = [k for k in d if not k.startswith("__")]
        return np.asarray(d[key if key else keys[0]])
    except NotImplementedError:
        import h5py

        with h5py.File(path, "r") as f:
            keys = [k for k in f.keys() if not k.startswith("#")]
            arr = np.asarray(f[key if key else keys[0]])
        return arr.T  # MATLAB is column-major


def _find_key(obj, name: str):
    """Recursive, case-insensitive search for a field in nested dicts/lists from loadmat."""
    lname = name.lower()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() == lname:
                return v
        for v in obj.values():
            r = _find_key(v, name)
            if r is not None:
                return r
    elif isinstance(obj, (list, tuple)) or (isinstance(obj, np.ndarray) and obj.dtype == object):
        for v in list(obj):
            r = _find_key(v, name)
            if r is not None:
                return r
    return None


def _find_tag(obj, tag_id: int):
    """Find a DNG tag value in MATLAB imfinfo 'UnknownTags' (struct array with ID/Value)."""
    if isinstance(obj, dict):
        if "ID" in obj and "Value" in obj and int(np.asarray(obj["ID"]).reshape(-1)[0]) == tag_id:
            return obj["Value"]
        for v in obj.values():
            r = _find_tag(v, tag_id)
            if r is not None:
                return r
    elif isinstance(obj, (list, tuple)) or (isinstance(obj, np.ndarray) and obj.dtype == object):
        for v in list(obj):
            r = _find_tag(v, tag_id)
            if r is not None:
                return r
    return None


def read_sidd_metadata(meta_path: str) -> dict:
    """Parse a SIDD *_METADATA_RAW_*.MAT file into plain numbers.

    Returns dict with as_shot_neutral, color_matrix1/2, noise_profile (or None),
    cfa_pattern (or None). Missing fields are left as None so callers can fall
    back to the per-camera configuration produced by scripts/check_rendering.py.
    """
    import scipy.io

    m = scipy.io.loadmat(meta_path, simplify_cells=True)
    m = m.get("metadata", m)
    out = {}
    asn = _find_key(m, "AsShotNeutral")
    out["as_shot_neutral"] = None if asn is None else np.asarray(asn, np.float64).reshape(-1)[:3]
    for i in (1, 2):
        cm = _find_key(m, f"ColorMatrix{i}")
        out[f"color_matrix{i}"] = None if cm is None else np.asarray(cm, np.float64).reshape(-1)[:9]
    npf = _find_key(m, "NoiseProfile")
    if npf is None:
        npf = _find_tag(m, 51041)
    out["noise_profile"] = None if npf is None else np.asarray(npf, np.float64).reshape(-1)
    cfa = _find_key(m, "CFAPattern2")
    if cfa is None:
        cfa = _find_tag(m, 33422)
    pat = None
    cfa_str = _find_key(m, "cfa_str")  # simplified metadata of some SIDD mirrors, e.g. 'grbg'
    if isinstance(cfa_str, str) and cfa_str.upper() in PATTERNS:
        pat = cfa_str.upper()
        cfa = None
    if cfa is not None:
        c = np.asarray(cfa).reshape(-1)[:4].astype(int)
        if set(c.tolist()) <= {0, 1, 2}:
            pat = "".join("RGB"[i] for i in c)
    out["cfa_pattern"] = pat if pat in PATTERNS else None
    return out


def read_sidd_nlf_csv(csv_path: str) -> dict:
    """noise_level_functions.csv -> {scene_instance_name: (a, b)} (best effort)."""
    table = {}
    if not os.path.exists(csv_path):
        return table
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 3:
                try:
                    table[row[0].strip()] = (float(row[1]), float(row[2]))
                except ValueError:
                    continue
    return table


def load_sidd(
    raw_path: str,
    meta_path: str,
    camera_cfg: dict | None = None,
    gt_srgb_path: str | None = None,
) -> tuple["RawImage", tuple[int, int]]:
    """Load a SIDD raw image (+ optional aligned GT sRGB).

    camera_cfg: per-camera rendering settings from configs/sidd_render.json
    (pattern, colour matrix choice). Returns (RawImage, (dy, dx)) where the
    offset is the crop applied to reach RGGB.
    """
    camera_cfg = camera_cfg or {}
    bayer = load_mat_array(raw_path).astype(np.float32)
    meta = read_sidd_metadata(meta_path)
    pattern = camera_cfg.get("pattern") or meta["cfa_pattern"] or "RGGB"
    bayer, (dy, dx) = to_rggb(np.clip(bayer, 0, 1), pattern)

    asn = meta["as_shot_neutral"]
    wb = wb_from_as_shot_neutral(asn) if asn is not None else None
    cm_key = camera_cfg.get("color_matrix", "color_matrix2")
    cm = meta.get(cm_key)
    if cm is None:
        cm = meta.get("color_matrix1")
    if wb is None or cm is None:
        raise InputError(f"SIDD metadata incomplete in {meta_path}")
    cam2srgb = cam2srgb_from_xyz2cam(cm, rownorm_xyz2cam=camera_cfg.get("rownorm", True))
    npf = meta["noise_profile"]
    if npf is not None and len(npf) >= 2:
        nlf = (float(npf[0]), float(npf[1]))
    else:
        from .bayer import pack_rggb

        nlf = estimate_nlf(pack_rggb(bayer))
    gt = None
    if gt_srgb_path:
        gt = cv2.cvtColor(cv2.imread(gt_srgb_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        gt = gt[dy : dy + bayer.shape[0], dx : dx + bayer.shape[1]]
    img = RawImage(bayer, wb, cam2srgb, nlf, "sidd", gt_srgb=gt, meta={"pattern": pattern, **camera_cfg})
    return img, (dy, dx)


# ----------------------------------------------------------------------- DNG
def load_dng(data: bytes | str) -> RawImage:
    import rawpy

    try:
        raw = rawpy.imread(io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data)
    except Exception as e:  # rawpy raises several libraw error types
        raise InputError(f"Could not decode the RAW file ({type(e).__name__}). Is it a valid DNG/RAW?") from e
    with raw:
        notes = []
        vis = raw.raw_image_visible
        if vis.ndim != 2 or raw.num_colors < 3 or raw.raw_pattern is None:
            raise InputError(
                "This RAW is not a Bayer mosaic (e.g. Apple ProRAW / linear DNG). "
                "Shoot in a Bayer RAW mode (Android Pro/RAW, Lightroom DNG) or upload a JPEG/PNG."
            )
        desc = raw.color_desc.decode()
        pattern = pattern_from_cfa(raw.raw_pattern, desc)
        black = np.asarray(raw.black_level_per_channel, np.float32)
        white = float(raw.white_level)
        # per-CFA-site black level
        bl = np.zeros(vis.shape, np.float32)
        cfa = raw.raw_colors_visible
        for i in range(4):
            bl[cfa == i] = black[i]
        bayer = np.clip((vis.astype(np.float32) - bl) / max(white - black.mean(), 1.0), 0, 1)
        cam_wb = np.asarray(raw.camera_whitebalance, np.float64)[:3]
        if np.all(cam_wb[:3] > 0):
            wb = (cam_wb / cam_wb[1]).astype(np.float32)
        else:
            wb = None
        xyz2cam = np.asarray(raw.rgb_xyz_matrix, np.float64)[:3, :3]
        flip = raw.sizes.flip
    bayer, _ = to_rggb(bayer, pattern)
    from .bayer import pack_rggb

    packed = pack_rggb(bayer)
    if wb is None:
        wb = wb_gray_world(packed)
        notes.append("No as-shot white balance in file: used gray-world estimate.")
    if not np.any(xyz2cam):
        cam2srgb = np.eye(3, dtype=np.float32)
        notes.append("No colour matrix in file: colour correction skipped (identity).")
    else:
        cam2srgb = cam2srgb_from_xyz2cam(xyz2cam)
    nlf = estimate_nlf(packed)
    notes.append(f"Noise level estimated from the image (a={nlf[0]:.2e}, b={nlf[1]:.2e}).")
    img = RawImage(bayer, wb, cam2srgb, nlf, "dng", meta={"pattern": pattern, "flip": flip}, notes=notes)
    return img


# ----------------------------------------------------------------------- sRGB inputs
def decode_heic_bytes(data: bytes) -> np.ndarray:
    """iPhone HEIC/HEIF -> RGB float in [0,1], rotated per EXIF like the phone shows it."""
    try:
        import pillow_heif
        from PIL import Image, ImageOps
    except ImportError as e:
        raise InputError("HEIC support is not installed on this server (pillow-heif). Upload a JPG instead.") from e
    pillow_heif.register_heif_opener()
    try:
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception as e:
        raise InputError(f"Could not decode the HEIC image ({type(e).__name__}).") from e
    return np.asarray(im, np.float32) / 255.0


def decode_image_bytes(data: bytes) -> np.ndarray:
    # IMREAD_COLOR applies the EXIF rotation (phone photos), ANYDEPTH keeps 16-bit PNG/TIFF precision
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
    if arr is None:
        raise InputError("Could not decode the image. Supported: DNG, JPG, PNG, TIFF, HEIC, SIDD .MAT.")
    scale = 65535.0 if arr.dtype == np.uint16 else 255.0
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB).astype(np.float32) / scale


def from_srgb(srgb: np.ndarray, noise_level: float = 0.5, seed: int = 0) -> RawImage:
    """Simulate a phone RAW capture of an sRGB image (with the original as ground truth)."""
    rng = np.random.default_rng(seed)
    u = unprocess(srgb, rng=rng, noise_level=noise_level)
    img = RawImage(
        u["bayer_noisy"], u["wb_gains"], u["cam2srgb"], u["nlf"], "synthetic", gt_srgb=u["clean_srgb"],
        meta={"noise_level": noise_level},
    )
    img.notes.append(
        "sRGB input: converted to a simulated noisy Bayer RAW (unprocessing, Brooks et al. 2019) "
        f"at noise level {noise_level:.2f}; the uploaded image is used as ground truth."
    )
    return img


# ----------------------------------------------------------------------- dispatch
RAW_EXT = {".dng", ".nef", ".cr2", ".cr3", ".arw", ".orf", ".rw2", ".raf"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
HEIC_EXT = {".heic", ".heif"}


def load_any(path: str, noise_level: float = 0.5, meta_path: str | None = None) -> RawImage:
    if path is None or not os.path.exists(path):
        raise InputError("No file received. Please upload an image.")
    size_mb = os.path.getsize(path) / 1e6
    if size_mb > MAX_FILE_MB:
        raise InputError(f"File is {size_mb:.0f} MB; the limit is {MAX_FILE_MB} MB.")
    ext = os.path.splitext(path)[1].lower()
    if ext in RAW_EXT:
        img = load_dng(path)
    elif ext == ".mat":
        if meta_path is None:
            raise InputError("SIDD .MAT raw needs its METADATA .MAT file; upload a DNG or JPG instead.")
        img, _ = load_sidd(path, meta_path)
    elif ext in IMG_EXT or ext in HEIC_EXT:
        with open(path, "rb") as f:
            data = f.read()
        rgb = decode_heic_bytes(data) if ext in HEIC_EXT else decode_image_bytes(data)
        img = from_srgb(rgb, noise_level=noise_level)
    else:
        raise InputError(f"Unsupported file type '{ext}'. Supported: DNG/RAW, JPG, PNG, TIFF, HEIC.")
    h, w = img.bayer.shape
    if min(h, w) < MIN_SIDE:
        raise InputError(f"Image is too small ({w}x{h}); minimum side is {MIN_SIDE} px.")
    return img
