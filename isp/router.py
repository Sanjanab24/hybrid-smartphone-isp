"""Adaptive-compute routing: give each tile only as much compute as it needs.

Each tile gets a difficulty score  d = mean noise std * (0.25 + detail / median detail),
where detail is gradient energy of a pre-smoothed luma. Tiles are ranked, and a
compute budget (fractions per path) sends the easiest tiles to the classical
guided filter, the hardest to the teacher, and the rest to the student. Tiles
overlap and are blended with feathered weights, so seams between paths are not
visible.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .color import rgb_to_luma
from .filters import guided_denoise

PATHS = ("classical", "student", "teacher")
# muted palette shared with the app (slate-indigo / sage / clay)
PATH_COLORS = {"classical": (0x6C / 255, 0x7A / 255, 0xA4 / 255), "student": (0x6E / 255, 0x9C / 255, 0x8A / 255),
               "teacher": (0xB9 / 255, 0x7A / 255, 0x68 / 255)}

# fractions of tiles per path (classical, student, teacher)
BUDGETS = {
    "classical_only": (1.0, 0.0, 0.0),
    "low": (0.5, 0.5, 0.0),
    "balanced": (0.25, 0.65, 0.10),
    "high": (0.0, 0.6, 0.4),
    "student_only": (0.0, 1.0, 0.0),
    "teacher_only": (0.0, 0.0, 1.0),
}


class ClassicalRestorer:
    """Guided filter whose strength follows the tile's noise map."""

    name = "classical"

    def __call__(self, x: np.ndarray) -> np.ndarray:  # (N, 6, H, W) -> (N, 3, H, W)
        out = np.empty((x.shape[0], 3, x.shape[2], x.shape[3]), np.float32)
        for i in range(x.shape[0]):
            img = x[i, :3].transpose(1, 2, 0)
            sigma = float(np.median(x[i, 3:]))
            out[i] = guided_denoise(img, sigma).transpose(2, 0, 1)
        return out


class OnnxRestorer:
    """Runs an exported restoration network with ONNX Runtime (CPU or CUDA)."""

    def __init__(self, path: str, name: str, threads: int | None = None, multiple: int = 16):
        import onnxruntime as ort

        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self.sess = ort.InferenceSession(path, so, providers=providers)
        self.inp = self.sess.get_inputs()[0].name
        self.name = name
        self.multiple = multiple

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h, w = x.shape[2:]
        ph, pw = (-h) % self.multiple, (-w) % self.multiple
        if ph or pw:
            x = np.pad(x, ((0, 0), (0, 0), (0, ph), (0, pw)), mode="reflect")
        y = self.sess.run(None, {self.inp: x.astype(np.float32)})[0]
        return np.clip(y[:, :, :h, :w], 0, 1)


@dataclass
class RouteResult:
    image: np.ndarray
    route_map: np.ndarray  # (rows, cols) index into PATHS
    tile_boxes: list
    fractions: dict
    timings: dict


def _positions(n: int, tile: int, stride: int) -> list[int]:
    if n <= tile:
        return [0]
    pos = list(range(0, n - tile, stride))
    pos.append(n - tile)
    return sorted(set(pos))


def _window(tile: int, overlap: int) -> np.ndarray:
    i = np.arange(tile, dtype=np.float32) + 0.5
    ramp = np.minimum(1.0, np.minimum(i, tile - i) / max(overlap, 1))
    w = np.maximum(ramp, 1e-3)
    return np.outer(w, w).astype(np.float32)


def tile_difficulty(srgb: np.ndarray, nmap: np.ndarray, boxes: list) -> np.ndarray:
    y = rgb_to_luma(cv2.GaussianBlur(srgb, (0, 0), 1.5))
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    nl = nmap.mean(-1)
    sig = np.array([nl[a:b, c:d].mean() for a, b, c, d in boxes])
    det = np.array([grad[a:b, c:d].mean() for a, b, c, d in boxes])
    det_n = det / (np.median(det) + 1e-6)
    return sig * (0.25 + det_n)


def assign_paths(scores: np.ndarray, fractions: tuple, available: dict, random_routing: bool = False,
                 seed: int = 0) -> np.ndarray:
    n = len(scores)
    order = np.random.default_rng(seed).permutation(n) if random_routing else np.argsort(scores, kind="stable")
    fc, fs, ft = fractions
    n_c = int(round(fc * n))
    n_t = int(round(ft * n))
    n_t = min(n_t, n - n_c)
    labels = np.full(n, 1, np.int64)
    labels[order[:n_c]] = 0
    if n_t:
        labels[order[n - n_t:]] = 2
    # fall back when a path has no model loaded: teacher -> student -> classical
    if not available.get("teacher"):
        labels[labels == 2] = 1
    if not available.get("student"):
        labels[labels == 1] = 2 if available.get("teacher") else 0
    return labels


def run_routed(srgb: np.ndarray, nmap: np.ndarray, restorers: dict, budget: str | tuple = "balanced",
               tile: int = 256, overlap: int = 32, batch: int = 8, random_routing: bool = False) -> RouteResult:
    """Restore a rendered sRGB image tile by tile with per-tile path selection."""
    fractions = BUDGETS[budget] if isinstance(budget, str) else tuple(budget)
    restorers = {"classical": restorers.get("classical") or ClassicalRestorer(), **{k: v for k, v in restorers.items() if k != "classical"}}
    H, W = srgb.shape[:2]
    ph, pw = max(0, tile - H), max(0, tile - W)
    if ph or pw:
        srgb_p = np.pad(srgb, ((0, ph), (0, pw), (0, 0)), mode="reflect")
        nmap_p = np.pad(nmap, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    else:
        srgb_p, nmap_p = srgb, nmap
    Hp, Wp = srgb_p.shape[:2]
    stride = tile - overlap
    ys, xs = _positions(Hp, tile, stride), _positions(Wp, tile, stride)
    boxes = [(y, y + tile, x, x + tile) for y in ys for x in xs]

    t0 = time.perf_counter()
    scores = tile_difficulty(srgb_p, nmap_p, boxes)
    labels = assign_paths(scores, fractions, {k: restorers.get(k) is not None for k in PATHS}, random_routing)
    timings = {"routing": time.perf_counter() - t0}

    inp = np.concatenate([srgb_p, nmap_p], -1).transpose(2, 0, 1).astype(np.float32)
    acc = np.zeros((3, Hp, Wp), np.float32)
    wsum = np.zeros((Hp, Wp), np.float32)
    win = _window(tile, overlap)
    for pi, path in enumerate(PATHS):
        idx = np.where(labels == pi)[0]
        if len(idx) == 0:
            continue
        fn = restorers[path]
        t0 = time.perf_counter()
        for s in range(0, len(idx), batch):
            chunk = idx[s : s + batch]
            x = np.stack([inp[:, a:b, c:d] for a, b, c, d in (boxes[i] for i in chunk)])
            y = fn(x)
            for j, i in enumerate(chunk):
                a, b, c, d = boxes[i]
                acc[:, a:b, c:d] += y[j] * win
                wsum[a:b, c:d] += win
        timings[path] = time.perf_counter() - t0
    out = (acc / np.maximum(wsum, 1e-6)).transpose(1, 2, 0)[:H, :W]
    fr = {p: float(np.mean(labels == i)) for i, p in enumerate(PATHS)}
    return RouteResult(np.clip(out, 0, 1), labels.reshape(len(ys), len(xs)), boxes, fr, timings)


def route_overlay(image: np.ndarray, res: RouteResult, alpha: float = 0.35) -> np.ndarray:
    """Colour each tile's core region by the path that processed it."""
    H, W = image.shape[:2]
    over = np.zeros((H, W, 3), np.float32)
    labels = res.route_map.reshape(-1)
    for (a, b, c, d), lab in zip(res.tile_boxes, labels):
        over[a:min(b, H), c:min(d, W)] = PATH_COLORS[PATHS[lab]]
    out = (1 - alpha) * image + alpha * over
    # grid lines
    for (a, b, c, d) in res.tile_boxes:
        if a < H:
            out[a, c:min(d, W)] = 1.0
        if c < W:
            out[a:min(b, H), c] = 1.0
    return np.clip(out, 0, 1)
