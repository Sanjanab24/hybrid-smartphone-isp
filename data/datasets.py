"""PyTorch datasets for the four training tasks.

render_first : noisy RAW -> classical render -> (sRGB, noise map) ; target GT sRGB   (teacher, student)
raw_first    : packed noisy RAW + packed noise std           ; target packed GT RAW (variant C)
e2e          : packed noisy RAW                              ; target GT sRGB       (variant B)
Sources: SIDD crop shards (data/sidd.py) or synthetic unprocessed sRGB images.
"""
from __future__ import annotations

import glob
import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from isp.bayer import flip_rggb, pack_rggb
from isp.noise import estimate_srgb_noise_map, raw_noise_std
from isp.pipeline import RenderConfig, render
from isp.raw_io import RawImage
from isp.unprocess import unprocess

MARGIN = 4  # raw pixels rendered around a crop and then discarded (demosaic borders)


def _chw(x):
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).float()


def make_sample(task: str, noisy: np.ndarray, gt_raw: np.ndarray | None, gt_srgb: np.ndarray | None,
                meta: dict, cfg: RenderConfig, est_map: bool, noisy_srgb: np.ndarray | None = None):
    """Build (input, target) for one crop. Arrays include MARGIN on every side."""
    m = MARGIN
    wb = np.asarray(meta["wb_gains"], np.float32)
    ccm = np.asarray(meta["cam2srgb"], np.float32)
    nlf = tuple(meta["nlf"])
    if task == "render_first":
        if meta.get("render"):  # per-camera settings found by scripts/check_rendering.py
            cfg = RenderConfig(**{**cfg.to_dict(), **meta["render"]})
        r = render(RawImage(noisy, wb, ccm, nlf, "train"), cfg)
        srgb = r.srgb if noisy_srgb is None else noisy_srgb
        nmap = estimate_srgb_noise_map(srgb) if est_map else r.noise_map
        inp = np.concatenate([srgb, nmap], -1)[m:-m, m:-m]
        return _chw(inp), _chw(gt_srgb[m:-m, m:-m])
    if task == "raw_first":
        pk = pack_rggb(noisy)
        inp = np.concatenate([pk, raw_noise_std(pk, *nlf)], -1)[m // 2:-m // 2, m // 2:-m // 2]
        return _chw(inp), _chw(pack_rggb(gt_raw)[m // 2:-m // 2, m // 2:-m // 2])
    if task == "e2e":
        return _chw(pack_rggb(noisy)[m // 2:-m // 2, m // 2:-m // 2]), _chw(gt_srgb[m:-m, m:-m])
    raise ValueError(task)


def _flip_all(arrs: list, fx: bool, fy: bool, size: int):
    """Flip mosaics (keeping RGGB) and aligned images identically; output side = size."""
    out = []
    for a in arrs:
        if a is None:
            out.append(None)
            continue
        if fx:
            a = a[:, ::-1]
        if fy:
            a = a[::-1]
        oy, ox = (1 if fy else 0), (1 if fx else 0)
        out.append(np.ascontiguousarray(a[oy:oy + size, ox:ox + size]))
    return out


class SiddShards(Dataset):
    def __init__(self, root: str, task: str, crop: int = 256, render_cfg: RenderConfig | None = None,
                 est_map_prob: float = 0.3, length: int | None = None, train: bool = True,
                 use_sidd_srgb: bool = False, seed: int = 0):
        self.dirs = sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.exists(os.path.join(d, "meta.json")))
        if not self.dirs:
            raise FileNotFoundError(f"no shards in {root} (run scripts/prepare_sidd.py)")
        self.task, self.crop, self.cfg = task, crop, render_cfg or RenderConfig()
        self.est_map_prob, self.train, self.use_sidd_srgb = est_map_prob, train, use_sidd_srgb
        self.metas = [json.load(open(os.path.join(d, "meta.json"))) for d in self.dirs]
        self.n_per = [np.load(os.path.join(d, "noisy_raw.npy"), mmap_mode="r").shape[0] for d in self.dirs]
        self.index = [(i, k) for i, n in enumerate(self.n_per) for k in range(n)]
        self.length = length or len(self.index)
        self.seed = seed
        self._cache = {}

    def __len__(self):
        return self.length

    def _arr(self, i, name):
        key = (i, name)
        if key not in self._cache:
            p = os.path.join(self.dirs[i], f"{name}.npy")
            self._cache[key] = np.load(p, mmap_mode="r") if os.path.exists(p) else None
        return self._cache[key]

    def __getitem__(self, idx):
        rng = np.random.default_rng(None if self.train else self.seed + idx)
        i, k = self.index[(rng.integers(len(self.index)) if self.train else idx % len(self.index))]
        meta = self.metas[i]
        full = self.crop + 2 * MARGIN
        size_with_flip = full + 2
        src = self._arr(i, "noisy_raw")
        H = src.shape[1]
        if self.train:
            y = int(rng.integers(0, (H - size_with_flip) // 2 + 1)) * 2
            x = int(rng.integers(0, (H - size_with_flip) // 2 + 1)) * 2
        else:
            y = x = ((H - size_with_flip) // 4) * 2
        sl = (k, slice(y, y + size_with_flip), slice(x, x + size_with_flip))
        noisy = np.asarray(src[sl], np.float32)
        gt_raw = np.asarray(self._arr(i, "gt_raw")[sl], np.float32) if self.task == "raw_first" else None
        g = self._arr(i, "gt_srgb")
        gt_srgb = np.asarray(g[sl], np.float32) / 255 if g is not None and self.task != "raw_first" else None
        ns = self._arr(i, "noisy_srgb") if self.use_sidd_srgb else None
        ns = np.asarray(ns[sl], np.float32) / 255 if ns is not None else None
        fx, fy = (bool(rng.integers(2)), bool(rng.integers(2))) if self.train else (False, False)
        noisy, gt_raw, gt_srgb, ns = _flip_all([noisy, gt_raw, gt_srgb, ns], fx, fy, full)
        est = self.train and rng.uniform() < self.est_map_prob
        return make_sample(self.task, noisy, gt_raw, gt_srgb, meta, self.cfg, est, ns)


class SyntheticUnprocessed(Dataset):
    """Clean sRGB images -> unprocess -> noisy RAW with random camera, WB and ISO.

    Used for pre-training / robustness and for smoke tests without SIDD.
    """

    def __init__(self, image_dir_or_arrays, task: str = "render_first", crop: int = 256,
                 render_cfg: RenderConfig | None = None, length: int = 1000, train: bool = True,
                 est_map_prob: float = 0.3, seed: int = 0):
        if isinstance(image_dir_or_arrays, str):
            files = sorted(f for ext in ("png", "jpg", "jpeg", "bmp")
                           for f in glob.glob(os.path.join(image_dir_or_arrays, "**", f"*.{ext}"), recursive=True))
            if not files:
                raise FileNotFoundError(f"no images in {image_dir_or_arrays}")
            self.files, self.arrays = files, None
        else:
            self.files, self.arrays = None, [a.astype(np.float32) for a in image_dir_or_arrays]
        self.task, self.crop, self.cfg, self.length = task, crop, render_cfg or RenderConfig(), length
        self.train, self.est_map_prob, self.seed = train, est_map_prob, seed

    def __len__(self):
        return self.length

    def _img(self, j):
        if self.arrays is not None:
            return self.arrays[j % len(self.arrays)]
        im = cv2.imread(self.files[j % len(self.files)], cv2.IMREAD_COLOR)
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255

    def __getitem__(self, idx):
        rng = np.random.default_rng(None if self.train else self.seed + idx)
        n = len(self.arrays) if self.arrays is not None else len(self.files)
        img = self._img(int(rng.integers(n)) if self.train else idx)
        full = self.crop + 2 * MARGIN
        h, w = img.shape[:2]
        if h < full or w < full:
            img = cv2.resize(img, (max(w, full), max(h, full)), interpolation=cv2.INTER_CUBIC)
            h, w = img.shape[:2]
        y = int(rng.integers(0, h - full + 1)) // 2 * 2
        x = int(rng.integers(0, w - full + 1)) // 2 * 2
        img = img[y:y + full, x:x + full]
        u = unprocess(img, rng=rng)
        meta = {"wb_gains": u["wb_gains"], "cam2srgb": u["cam2srgb"], "nlf": u["nlf"]}
        est = self.train and rng.uniform() < self.est_map_prob
        return make_sample(self.task, u["bayer_noisy"], u["bayer_clean"], u["clean_srgb"], meta, self.cfg, est)
