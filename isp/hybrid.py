"""The full hybrid ISP: classical rendering -> routed learned restoration -> adaptive post.

Used identically by the app (ONNX restorers) and the evaluation (torch or ONNX).
"""
from __future__ import annotations

import time

import numpy as np

from .filters import adaptive_unsharp
from .pipeline import RenderConfig, render
from .raw_io import RawImage
from .router import run_routed


class HybridISP:
    def __init__(self, restorers: dict, render_cfg: RenderConfig | None = None, tile: int = 256,
                 overlap: int = 32, post_sharpen: bool = False):
        self.restorers = restorers  # {"student": callable, "teacher": callable} (classical is built in)
        self.cfg = render_cfg or RenderConfig()
        self.tile, self.overlap, self.post_sharpen = tile, overlap, post_sharpen

    def available(self) -> dict:
        return {k: self.restorers.get(k) is not None for k in ("student", "teacher")}

    def process(self, raw: RawImage, budget="balanced", keep_stages: bool = False, random_routing: bool = False,
                srgb_override: np.ndarray | None = None):
        """srgb_override: use this rendered sRGB (e.g. SIDD's own rendering) instead of ours,
        keeping our propagated noise map. Used for SIDD evaluation to match training."""
        t_all = time.perf_counter()
        r = render(raw, self.cfg, keep_stages)
        if srgb_override is not None:
            h, w = r.srgb.shape[:2]
            r.srgb = np.ascontiguousarray(srgb_override[:h, :w], dtype=np.float32)
        res = run_routed(r.srgb, r.noise_map, self.restorers, budget, self.tile, self.overlap,
                         random_routing=random_routing)
        out = res.image
        timings = {**r.timings, **{f"restore_{k}": v for k, v in res.timings.items()}}
        if self.post_sharpen:
            t0 = time.perf_counter()
            # residual noise after learned restoration is roughly 10% of the input noise
            out = adaptive_unsharp(out, float(np.mean(r.noise_map)) * 0.1, base_amount=0.4)
            timings["sharpen"] = time.perf_counter() - t0
        timings["total"] = time.perf_counter() - t_all
        if keep_stages:
            r.stages["6 Learned restoration (routed)"] = res.image
        return {"output": out, "rendered": r, "route": res, "timings": timings}
