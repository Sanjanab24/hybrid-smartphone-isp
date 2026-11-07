"""SIDD indexing and preprocessing into crop shards.

SIDD-Medium layout (folder names are kept by the Kaggle mirrors too):
  <raw_root>/**/0001_001_S6_00100_00060_3200_L/0001_NOISY_RAW_010.MAT
                                               0001_GT_RAW_010.MAT
                                               0001_METADATA_RAW_010.MAT
  <srgb_root>/**/0001_001_S6_00100_00060_3200_L/0001_NOISY_SRGB_010.PNG
                                                0001_GT_SRGB_010.PNG
Folder fields: instance_scene_camera_ISO_shutter_illuminant_brightness.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import cv2
import numpy as np

from isp.bayer import to_rggb
from isp.raw_io import load_sidd

CAMERAS = ("GP", "IP", "S6", "N6", "G4")


@dataclass
class SiddItem:
    key: str  # e.g. 0001_001_S6_00100_00060_3200_L/010
    folder: str
    camera: str
    iso: int
    noisy_raw: str
    gt_raw: str
    meta: str
    noisy_srgb: str | None
    gt_srgb: str | None


def parse_folder(name: str) -> dict:
    p = name.split("_")
    return {"instance": p[0], "scene": p[1], "camera": p[2], "iso": int(p[3]), "shutter": int(p[4]),
            "illuminant": p[5], "brightness": p[6] if len(p) > 6 else ""}


def index_sidd(raw_root: str, srgb_root: str | None = None) -> list[SiddItem]:
    srgb_dirs = {}
    if srgb_root:
        for d in glob.glob(os.path.join(srgb_root, "**", "*_*_*_*_*_*_*"), recursive=True):
            if os.path.isdir(d):
                srgb_dirs[os.path.basename(d)] = d
    items = []
    for noisy in sorted(glob.glob(os.path.join(raw_root, "**", "*_NOISY_RAW_*.MAT"), recursive=True)):
        folder = os.path.basename(os.path.dirname(noisy))
        info = parse_folder(folder)
        fname = os.path.basename(noisy)
        inst, _, _, suffix = fname.split("_", 3)  # 0001 NOISY RAW 010.MAT
        suffix = suffix.split(".")[0]
        d = os.path.dirname(noisy)
        gt = os.path.join(d, f"{inst}_GT_RAW_{suffix}.MAT")
        meta = os.path.join(d, f"{inst}_METADATA_RAW_{suffix}.MAT")
        if not (os.path.exists(gt) and os.path.exists(meta)):
            continue
        ns = gs = None
        if folder in srgb_dirs:
            sd = srgb_dirs[folder]
            ns = os.path.join(sd, f"{inst}_NOISY_SRGB_{suffix}.PNG")
            gs = os.path.join(sd, f"{inst}_GT_SRGB_{suffix}.PNG")
            ns = ns if os.path.exists(ns) else None
            gs = gs if os.path.exists(gs) else None
        items.append(SiddItem(f"{folder}/{suffix}", folder, info["camera"], info["iso"], noisy, gt, meta, ns, gs))
    return items


def split_items(items: list[SiddItem], test_camera: str | None, val_frac: float = 0.08, seed: int = 0):
    """Leave-one-camera-out test split; validation = random scene instances of the other cameras."""
    test = [it for it in items if test_camera and it.camera == test_camera]
    rest = [it for it in items if not (test_camera and it.camera == test_camera)]
    folders = sorted({it.folder for it in rest})
    rng = np.random.default_rng(seed)
    val_f = set(rng.choice(folders, size=max(1, int(len(folders) * val_frac)), replace=False)) if folders else set()
    val = [it for it in rest if it.folder in val_f]
    train = [it for it in rest if it.folder not in val_f]
    return train, val, test


def _read_png(path):
    return cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def load_camera_cfg(path: str = "configs/sidd_render.json") -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def write_item_crops(it: SiddItem, out_dir: str, n_crops: int, crop: int, cam_cfg: dict, rng: np.random.Generator):
    """Store aligned crops of one SIDD image as .npy (memory-mappable) + meta.json."""
    raw, (dy, dx) = load_sidd(it.noisy_raw, it.meta, cam_cfg.get(it.camera, cam_cfg.get("default")))
    from isp.raw_io import load_mat_array

    gt_raw = np.clip(load_mat_array(it.gt_raw).astype(np.float32), 0, 1)
    gt_raw = gt_raw[dy:, dx:][: raw.bayer.shape[0], : raw.bayer.shape[1]]
    h, w = raw.bayer.shape
    gs = _read_png(it.gt_srgb)[dy : dy + h, dx : dx + w] if it.gt_srgb else None
    ns = _read_png(it.noisy_srgb)[dy : dy + h, dx : dx + w] if it.noisy_srgb else None
    if gs is not None and gs.shape[:2] != (h, w):
        raise RuntimeError(f"{it.key}: sRGB shape {gs.shape} != raw {raw.bayer.shape}; check orientation config")
    ys = rng.integers(0, (h - crop) // 2, n_crops) * 2
    xs = rng.integers(0, (w - crop) // 2, n_crops) * 2
    d = os.path.join(out_dir, it.key.replace("/", "_"))
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "noisy_raw.npy"), np.stack([raw.bayer[y:y + crop, x:x + crop] for y, x in zip(ys, xs)]).astype(np.float16))
    np.save(os.path.join(d, "gt_raw.npy"), np.stack([gt_raw[y:y + crop, x:x + crop] for y, x in zip(ys, xs)]).astype(np.float16))
    if gs is not None:
        np.save(os.path.join(d, "gt_srgb.npy"), np.stack([gs[y:y + crop, x:x + crop] for y, x in zip(ys, xs)]))
    if ns is not None:
        np.save(os.path.join(d, "noisy_srgb.npy"), np.stack([ns[y:y + crop, x:x + crop] for y, x in zip(ys, xs)]))
    cc = cam_cfg.get(it.camera, cam_cfg.get("default", {})) or {}
    meta = {"key": it.key, "camera": it.camera, "iso": it.iso, "wb_gains": raw.wb_gains.tolist(),
            "cam2srgb": raw.cam2srgb.tolist(), "nlf": list(raw.nlf), "crop": crop,
            "render": cc.get("render", {})}
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f)
