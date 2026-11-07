"""Classical rendering front-end and the traditional ISP baseline.

render():  RAW -> (noisy) sRGB + propagated noise map + intermediate stages.
           Stage order matches the SIDD simple-camera-pipeline so the output
           matches the domain the pretrained restoration network was trained on.
traditional_isp(): render() + classical denoise + adaptive sharpening (baseline A).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .bayer import apply_wb_mosaic, bayer_visualization, pack_rggb
from .color import apply_ccm, auto_exposure_gain, gamma_encode
from .demosaic import demosaic
from .filters import adaptive_unsharp, bilateral_denoise, guided_denoise, nlm_denoise
from .noise import srgb_noise_map
from .raw_io import RawImage


@dataclass
class RenderConfig:
    demosaic: str = "mhc"  # bilinear | mhc | ea | vng
    gamma: str = "pow2.2"  # pow2.2 (SIDD) | srgb
    auto_exposure: bool = False  # adaptive digital gain for dark captures (off for SIDD eval)
    clip_wb: bool = True  # clip after WB like SIDD's pipeline

    def to_dict(self):
        return asdict(self)


@dataclass
class Rendered:
    srgb: np.ndarray  # (H, W, 3) gamma-encoded, in [0,1]
    noise_map: np.ndarray  # (H, W, 3) noise std in sRGB units
    linear: np.ndarray  # (H, W, 3) linear sRGB before gamma
    stages: dict  # name -> (H, W, 3) image for visualisation
    timings: dict  # stage -> seconds
    exposure_gain: float


def render(raw: RawImage, cfg: RenderConfig | None = None, keep_stages: bool = False) -> Rendered:
    cfg = cfg or RenderConfig()
    if raw.meta.get("render"):  # per-camera settings (configs/sidd_render.json) take precedence
        cfg = RenderConfig(**{**cfg.to_dict(), **raw.meta["render"]})
    t = {}
    stages = {}
    t0 = time.perf_counter()
    bayer = raw.bayer
    if keep_stages:
        stages["1 RAW mosaic (false colour)"] = bayer_visualization(bayer)

    wb = apply_wb_mosaic(bayer, raw.wb_gains)
    if cfg.clip_wb:
        wb = np.clip(wb, 0, 1)
    t["white_balance"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    rgb = demosaic(wb, cfg.demosaic)
    t["demosaic"] = time.perf_counter() - t0
    if keep_stages:
        stages["2 White balance + demosaic"] = np.clip(rgb, 0, 1) ** (1 / 2.2)

    t0 = time.perf_counter()
    lin = np.clip(apply_ccm(rgb, raw.cam2srgb), 0, 1)
    gain = raw.exposure_gain
    if cfg.auto_exposure:
        gain *= auto_exposure_gain(lin)
    if gain != 1.0:
        lin = np.clip(lin * gain, 0, 1)
    t["ccm"] = time.perf_counter() - t0
    if keep_stages:
        stages["3 Colour correction (CCM)"] = lin ** (1 / 2.2)

    t0 = time.perf_counter()
    srgb = gamma_encode(lin, cfg.gamma).astype(np.float32)
    t["gamma"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    lin_smooth = cv2.GaussianBlur(lin, (0, 0), 2.0)
    nmap = srgb_noise_map(pack_rggb(bayer), raw.nlf[0], raw.nlf[1], raw.wb_gains, raw.cam2srgb, lin_smooth,
                          cfg.gamma, exposure_gain=gain)
    t["noise_map"] = time.perf_counter() - t0
    if keep_stages:
        stages["4 Gamma / tone (noisy sRGB)"] = srgb
        stages["5 Noise map (x8)"] = np.clip(nmap * 8, 0, 1)
    return Rendered(srgb, nmap, lin, stages, t, gain)


def traditional_isp(raw: RawImage, cfg: RenderConfig | None = None, denoiser: str = "guided",
                    sharpen: bool = True, keep_stages: bool = False):
    """Baseline A: classical rendering + classical denoise + adaptive sharpening.

    The guided filter was the strongest classical denoiser in our tuning
    (beats bilateral and OpenCV NLM by 2-3 dB on unprocessed test images).
    """
    r = render(raw, cfg, keep_stages)
    sigma = float(np.mean(r.noise_map))
    t0 = time.perf_counter()
    fn = {"nlm": nlm_denoise, "bilateral": bilateral_denoise, "guided": guided_denoise}[denoiser]
    out = fn(r.srgb, sigma)
    r.timings["denoise_classical"] = time.perf_counter() - t0
    if sharpen:
        t0 = time.perf_counter()
        out = adaptive_unsharp(out, sigma * 0.3)
        r.timings["sharpen"] = time.perf_counter() - t0
    return out, r
