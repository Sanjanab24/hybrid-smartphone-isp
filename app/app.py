"""Hybrid Smartphone ISP - live demo (Gradio).

Upload a phone DNG (Bayer RAW) or any JPEG/PNG (converted to a simulated noisy
RAW so a ground truth exists). The app shows every ISP stage, traditional vs
hybrid output, which compute path each tile took, metrics and timings.

Runs on CPU with ONNX Runtime; no PyTorch needed. Models are read from
./weights (or downloaded from the Hugging Face repo in $WEIGHTS_REPO).
Without any model it still runs the classical path and says so.
"""
from __future__ import annotations

import glob
import logging
import os
import sys
import time
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import gradio as gr  # noqa: E402

from isp.hybrid import HybridISP  # noqa: E402
from isp.metrics import all_metrics  # noqa: E402
from isp.pipeline import RenderConfig, traditional_isp  # noqa: E402
from isp.raw_io import InputError, load_any  # noqa: E402
from isp.router import PATH_COLORS, OnnxRestorer, route_overlay  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("isp-app")

WEIGHTS = os.path.join(ROOT, "weights")
MAX_PIXELS = int(float(os.environ.get("MAX_MEGAPIXELS", "2.0")) * 1e6)
THREADS = int(os.environ.get("ORT_THREADS", "0")) or None
CANDIDATES = {
    "student": ["student_int8.onnx", "student_fp32.onnx"],
    "teacher": ["teacher_w32_fp32.onnx", "teacher_fp32.onnx"],
}


def _fetch(fname: str) -> str | None:
    path = os.path.join(WEIGHTS, fname)
    if os.path.exists(path):
        return path
    repo = os.environ.get("WEIGHTS_REPO")
    if not repo:
        return None
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo, fname, local_dir=WEIGHTS)
    except Exception as e:  # missing file in repo / offline
        log.info("weights %s not available from %s: %s", fname, repo, e)
        return None


def load_restorers() -> tuple[dict, dict]:
    rest, names = {}, {}
    for role, files in CANDIDATES.items():
        for f in files:
            p = _fetch(f)
            if p:
                try:
                    rest[role] = OnnxRestorer(p, role, threads=THREADS)
                    names[role] = f
                    break
                except Exception as e:
                    log.warning("could not load %s: %s", p, e)
    log.info("loaded restorers: %s", names)
    return rest, names


RESTORERS, MODEL_NAMES = load_restorers()
ISP = HybridISP(RESTORERS, RenderConfig())

BUDGET_UI = {
    "Fast": (0.5, 0.5, 0.0),
    "Balanced": (0.2, 0.7, 0.1),
    "Max quality": (0.0, 0.5, 0.5),
}


def _u8(x):
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def zoom_box(noise_map: np.ndarray, size: int = 192) -> tuple[int, int, int]:
    """Pick the square region where the noise is most visible (highest mean noise std)."""
    import cv2

    h, w = noise_map.shape[:2]
    s = int(min(size, h, w))
    m = cv2.blur(noise_map.mean(-1), (s, s))
    half = s // 2
    inner = m[half:h - s + half + 1, half:w - s + half + 1]
    if inner.size == 0:
        return 0, 0, s
    y, x = np.unravel_index(np.argmax(inner), inner.shape)
    return int(y), int(x), s


def zoom(img: np.ndarray, box, scale: int = 2) -> np.ndarray:
    import cv2

    y, x, s = box
    crop = img[y:y + s, x:x + s]
    return cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale), interpolation=cv2.INTER_NEAREST)


def process(file, noise_level, budget_name, full_res, progress=gr.Progress()):
    t_start = time.perf_counter()
    if file is None:
        raise gr.Error("Please upload an image (DNG RAW, JPG or PNG) or pick an example.")
    path = file if isinstance(file, str) else getattr(file, "name", None)
    try:
        progress(0.05, desc="Decoding input")
        raw = load_any(path, noise_level=float(noise_level))
        if not full_res:
            raw = raw.center_crop_pixels(MAX_PIXELS)
        elif raw.bayer.size > 4 * MAX_PIXELS:
            raw = raw.center_crop_pixels(4 * MAX_PIXELS)
        progress(0.25, desc="Traditional ISP baseline")
        t0 = time.perf_counter()
        trad, _ = traditional_isp(raw)
        t_trad = time.perf_counter() - t0
        progress(0.5, desc="Hybrid ISP (routed learned restoration)")
        res = ISP.process(raw, BUDGET_UI[budget_name], keep_stages=True)
    except InputError as e:
        raise gr.Error(str(e))
    except gr.Error:
        raise
    except MemoryError:
        raise gr.Error("Image too large for this server. Untick 'Full resolution' or upload a smaller file.")
    except Exception as e:
        log.error("processing failed: %s", traceback.format_exc())
        raise gr.Error(f"Processing failed ({type(e).__name__}: {e}). Try another file.")

    progress(0.9, desc="Metrics")
    out = res["output"]
    route = res["route"]
    noisy = res["rendered"].srgb
    h, w = raw.bayer.shape
    t = res["timings"]
    fr = route.fractions
    total = time.perf_counter() - t_start

    stages = [(_u8(v), k[2:] if k[:1].isdigit() else k) for k, v in res["rendered"].stages.items()]
    stages.append((_u8(out), "Final hybrid output"))
    overlay = route_overlay(out, route)
    box = zoom_box(res["rendered"].noise_map)
    zooms = [_u8(zoom(noisy, box)), _u8(zoom(trad, box)), _u8(zoom(out, box))]

    if raw.gt_srgb is not None:
        gt = raw.gt_srgb[: out.shape[0], : out.shape[1]]
        ms = {name: all_metrics(img[8:-8, 8:-8], gt[8:-8, 8:-8])
              for name, img in (("Noisy input", noisy), ("Traditional ISP", trad), ("Hybrid ISP (ours)", out))}
        summary = results_card(ms, w * h, total)
    else:
        summary = raw_card(w * h, total)

    models = ", ".join(f"{k}: <code>{v}</code>" for k, v in MODEL_NAMES.items()) or \
        "<b>none loaded</b> - classical path only"
    render_s = sum(t.get(k, 0) for k in ("white_balance", "demosaic", "ccm", "gamma", "noise_map"))
    restore_s = sum(v for k, v in t.items() if k.startswith("restore_"))
    details = (
        "| | |\n|---|---|\n"
        f"| Input | {raw.source.upper()}, {w}x{h} ({w * h / 1e6:.2f} MP) |\n"
        f"| Noise model | variance = {raw.nlf[0]:.2e} * x + {raw.nlf[1]:.2e} |\n"
        f"| Models | {models} |\n"
        f"| Compute split | classical {fr['classical']:.0%}, student {fr['student']:.0%}, teacher {fr['teacher']:.0%} |\n"
        f"| Latency | render {render_s:.2f} s, learned restoration {restore_s:.2f} s, "
        f"traditional baseline {t_trad:.2f} s, total {total:.2f} s |\n"
    )
    if raw.notes:
        details += "\n" + "\n".join(f"- {n}" for n in raw.notes)
    progress(1.0)
    slider = (_u8(noisy), _u8(out)) if HAS_SLIDER else _u8(out)
    full = [_u8(noisy), _u8(trad), _u8(out)]
    return (slider, summary, *zooms, _u8(overlay), stages, details, *full)


def results_card(ms: dict, pixels: int, seconds: float) -> str:
    """Stat tiles + metrics table; the best value in each column is highlighted."""
    names = list(ms)
    best = {"psnr": max(names, key=lambda n: ms[n]["psnr"]), "ssim": max(names, key=lambda n: ms[n]["ssim"]),
            "delta_e": min(names, key=lambda n: ms[n]["delta_e"])}
    hyb, trad = ms["Hybrid ISP (ours)"], ms["Traditional ISP"]
    d_psnr = hyb["psnr"] - trad["psnr"]
    d_de = (trad["delta_e"] - hyb["delta_e"]) / max(trad["delta_e"], 1e-6) * 100
    tiles = (
        stat_tile("sage", f"{d_psnr:+.1f} dB", "PSNR vs traditional ISP")
        + stat_tile("sand", f"{-d_de:+.0f}%", "colour error (Delta E) vs traditional")
        + stat_tile("slate", f"{seconds:.1f} s", f"to process {pixels / 1e6:.1f} MP on CPU")
    )
    rows = ""
    for n in names:
        cells = "".join(
            f"<td><span class='{'best' if best[k] == n else ''}'>{ms[n][k]:{fmt}}</span></td>"
            for k, fmt in (("psnr", ".2f"), ("ssim", ".3f"), ("delta_e", ".2f")))
        rows += f"<tr class='{'ours' if 'ours' in n else ''}'><td>{n}</td>{cells}</tr>"
    return (
        f"<div class='tiles'>{tiles}</div>"
        "<div class='card'><table class='mt'><thead><tr><th>Pipeline</th><th>PSNR (dB) &uarr;</th>"
        "<th>SSIM &uarr;</th><th>Delta E 2000 &darr;</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        "<p class='note'>Reference = the photo you uploaded. We simulate the noisy sensor capture, then restore it.</p>"
        "</div>"
    )


def stat_tile(tone: str, value: str, label: str) -> str:
    return f"<div class='tile {tone}'><div class='v'>{value}</div><div class='l'>{label}</div></div>"


def raw_card(pixels: int, seconds: float) -> str:
    return (
        "<div class='tiles'>"
        + stat_tile("slate", f"{seconds:.1f} s", f"to process {pixels / 1e6:.1f} MP on CPU")
        + stat_tile("sage", "RAW", "real sensor data, no reference image")
        + "</div><div class='card'><p class='note'>Compare visually with the slider above and the "
        "zoomed view below.</p></div>"
    )


HAS_SLIDER = hasattr(gr, "ImageSlider")
EXAMPLES = [e for e in sorted(glob.glob(os.path.join(ROOT, "app", "examples", "*.*")))
            if os.path.splitext(e)[1].lower() not in (".md", ".txt")]


def make_thumbnails(paths: list[str]) -> list[tuple[str | None, str]]:
    """Small preview images for the example picker (HEIC/DNG are decoded once at startup)."""
    import io as _io
    import tempfile

    import cv2

    from isp.raw_io import HEIC_EXT, decode_heic_bytes, decode_image_bytes

    out_dir = os.path.join(tempfile.gettempdir(), "hybrid_isp_thumbs")
    os.makedirs(out_dir, exist_ok=True)
    thumbs = []
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        try:
            with open(p, "rb") as f:
                data = f.read()
            if ext in HEIC_EXT:
                rgb = decode_heic_bytes(data)
            elif ext in (".dng", ".nef", ".cr2", ".arw"):
                import rawpy

                with rawpy.imread(_io.BytesIO(data)) as r:
                    rgb = r.postprocess(half_size=True, use_camera_wb=True).astype(np.float32) / 255
            else:
                rgb = decode_image_bytes(data)
            hh, ww = rgb.shape[:2]
            sc = 360 / max(hh, ww)
            small = cv2.resize(rgb, (max(1, int(ww * sc)), max(1, int(hh * sc))), interpolation=cv2.INTER_AREA)
            tp = os.path.join(out_dir, os.path.basename(p) + ".jpg")
            cv2.imwrite(tp, cv2.cvtColor(_u8(small), cv2.COLOR_RGB2BGR))
            thumbs.append((tp, os.path.splitext(os.path.basename(p))[0][:18]))
        except Exception as e:  # a broken example must not break the app
            log.warning("no thumbnail for %s: %s", p, e)
            thumbs.append((None, os.path.basename(p)))
    return thumbs


THUMBS = make_thumbnails(EXAMPLES)
SHOWN = [(t, c, p) for (t, c), p in zip(THUMBS, EXAMPLES) if t]


# ------------------------------------------------------------------ look and feel
# One muted palette ("dusk slate"): slate-indigo primary, sage / sand / clay accents, cool greys.
def _scale(name, *hexes):
    keys = ["c50", "c100", "c200", "c300", "c400", "c500", "c600", "c700", "c800", "c900", "c950"]
    return gr.themes.Color(name=name, **dict(zip(keys, hexes)))


SLATE = _scale("slate_indigo", "#F2F4F9", "#E6E9F2", "#CDD3E4", "#AEB8D2", "#8C99BD", "#6C7AA4", "#56648C",
               "#46537A", "#364160", "#2A334B", "#1C2233")
SAGE = _scale("sage", "#F1F6F3", "#E4EFEA", "#C7DDD3", "#A6C8B9", "#86B2A0", "#6E9C8A", "#587F70", "#476659",
              "#384F46", "#2A3B34", "#1B2622")
MIST = _scale("mist", "#F8F9FB", "#F1F3F7", "#E3E7EE", "#CBD1DC", "#A3ABBB", "#7C8597", "#5E6779", "#474F5F",
              "#2F3542", "#1F2430", "#14181F")

THEME = gr.themes.Base(
    primary_hue=SLATE, secondary_hue=SAGE, neutral_hue=MIST,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    radius_size=gr.themes.sizes.radius_lg, spacing_size=gr.themes.sizes.spacing_lg,
    text_size=gr.themes.sizes.text_md,
).set(
    body_background_fill="#F4F5F8", body_background_fill_dark="#11141B",
    body_text_color="#1F2633", body_text_color_dark="#E4E7EE",
    body_text_color_subdued="#6B7487", body_text_color_subdued_dark="#98A1B3",
    background_fill_primary="#FFFFFF", background_fill_primary_dark="#191D27",
    background_fill_secondary="#F8F9FB", background_fill_secondary_dark="#1E2330",
    block_background_fill="#FFFFFF", block_background_fill_dark="#191D27",
    block_border_color="#E3E7EE", block_border_color_dark="#2A3141",
    block_border_width="1px", block_shadow="0 1px 2px rgba(31,38,51,.04)", block_shadow_dark="none",
    block_label_background_fill="#F2F4F9", block_label_background_fill_dark="#232838",
    block_label_text_color="#56648C", block_label_text_color_dark="#AEB8D2",
    block_title_text_color="#3A4458", block_title_text_color_dark="#CDD3E4", block_title_text_weight="600",
    border_color_primary="#E3E7EE", border_color_primary_dark="#2A3141",
    input_background_fill="#F8F9FB", input_background_fill_dark="#1E2330",
    input_border_color="#E3E7EE", input_border_color_dark="#2A3141",
    button_primary_background_fill="#56648C", button_primary_background_fill_dark="#6C7AA4",
    button_primary_background_fill_hover="#46537A", button_primary_background_fill_hover_dark="#7C8AB4",
    button_primary_text_color="#FFFFFF", button_primary_border_color="#56648C",
    button_large_radius="12px",
    slider_color="#6E9C8A", slider_color_dark="#86B2A0",
    color_accent="#6E9C8A", color_accent_soft="#E4EFEA", color_accent_soft_dark="#24332D",
    checkbox_label_background_fill_selected="#E6E9F2", checkbox_label_background_fill_selected_dark="#2A334B",
    link_text_color="#56648C", link_text_color_dark="#AEB8D2",
    layout_gap="22px",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@600;700;800&display=swap');
:root {
  --ink: #1F2633; --muted: #6B7487; --line: #E3E7EE; --surface: #FFFFFF; --soft: #F8F9FB;
  --slate: #56648C; --slate-soft: #ECEFF6; --sage: #5E8E7B; --sage-soft: #E6F0EB;
  --sand: #A8843F; --sand-soft: #F6EFE1; --clay: #A9685A; --clay-soft: #F5E7E3;
  --accent: #4F63C9; --accent-soft: #E8EBFA;
}
.dark {
  --ink: #E4E7EE; --muted: #98A1B3; --line: #2A3141; --surface: #191D27; --soft: #1E2330;
  --slate: #AEB8D2; --slate-soft: #232A3C; --sage: #9CC4B3; --sage-soft: #1F2E28;
  --sand: #D8BC86; --sand-soft: #2E2919; --clay: #D59A8C; --clay-soft: #33231F;
  --accent: #9DAEF5; --accent-soft: #252B47;
}
.gradio-container {max-width: 1240px !important; margin: 0 auto !important; padding: 28px 28px 48px !important;}
h1, h2, h3, .hero .title {font-family: 'Manrope', 'Inter', sans-serif !important; letter-spacing: -0.01em;}

/* header */
.hero {padding: 6px 2px 18px; border-bottom: 1px solid var(--line); margin-bottom: 6px;}
.hero .title {font-size: 34px; font-weight: 800; color: var(--ink); margin: 0 0 6px;}
.hero .subtitle {font-size: 15.5px; color: var(--muted); max-width: 760px; line-height: 1.55;}
.chips {display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px;}
.chip {font-size: 12.5px; font-weight: 600; padding: 5px 12px; border-radius: 999px;}
.chip.slate {background: var(--slate-soft); color: var(--slate);}
.chip.sage {background: var(--sage-soft); color: var(--sage);}
.chip.sand {background: var(--sand-soft); color: var(--sand);}
.chip.clay {background: var(--clay-soft); color: var(--clay);}

/* step headings */
.step {font-family: 'Manrope', sans-serif; font-size: 16.5px; font-weight: 700; color: var(--accent);
  display: flex; align-items: center; margin: 6px 0 -8px;}
.step .n {display: inline-flex; width: 22px; height: 22px; border-radius: 50%; align-items: center;
  justify-content: center; font-size: 12.5px; margin-right: 9px; background: var(--accent-soft); color: var(--accent);}
.fullrow {margin-top: 4px;}

/* results */
.tiles {display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 14px;}
.tile {border-radius: 14px; padding: 14px 16px; border: 1px solid var(--line);}
.tile .v {font-family: 'Manrope', sans-serif; font-size: 26px; font-weight: 800; line-height: 1.1;}
.tile .l {font-size: 12.5px; color: var(--muted); margin-top: 4px;}
.tile.sage {background: var(--sage-soft);} .tile.sage .v {color: var(--sage);}
.tile.sand {background: var(--sand-soft);} .tile.sand .v {color: var(--sand);}
.tile.slate {background: var(--slate-soft);} .tile.slate .v {color: var(--slate);}
.card {background: var(--surface); border: 1px solid var(--line); border-radius: 14px; padding: 6px 16px 12px;}
.card .note, p.note {font-size: 12.5px; color: var(--muted); margin: 10px 0 0;}
.placeholder {text-align: center; padding: 26px 16px; color: var(--muted); border: 1px dashed var(--line);
  border-radius: 14px; font-size: 14px;}
table.mt {width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 14px;}
table.mt th {font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: .04em; color: var(--muted);
  padding: 12px 8px 8px; text-align: right; border-bottom: 1px solid var(--line);}
table.mt td {padding: 10px 8px; text-align: right; border-bottom: 1px solid var(--line); color: var(--ink);}
table.mt tr:last-child td {border-bottom: none;}
table.mt, table.mt tr, table.mt th, table.mt td {border-left: none !important; border-right: none !important;
  border-top: none !important; background: transparent !important;}
table.mt {border: none !important; margin: 0 !important;}
table.mt th:first-child, table.mt td:first-child {text-align: left;}
table.mt tr.ours td:first-child {font-weight: 700; color: var(--slate);}
table.mt span.best {background: var(--sage-soft); color: var(--sage); font-weight: 700; padding: 2px 8px; border-radius: 6px;}

/* route legend */
.legend {display: flex; gap: 8px; flex-wrap: wrap; margin: 4px 0 8px;}
.legend span {font-size: 12.5px; font-weight: 600; padding: 4px 12px; border-radius: 999px; color: #fff;}
.muted {color: var(--muted); font-size: 14px; line-height: 1.55;}
.tabs-wrap {margin-top: 10px;}
footer {opacity: .6;}
"""

HEADER = """
<div class='hero'>
  <div class='title'>Hybrid Smartphone ISP</div>
  <div class='subtitle'>A classical camera pipeline (white balance, demosaic, colour correction, tone) combined with a
  compact learned denoiser. Upload a phone photo or a RAW file and compare it against a traditional ISP.</div>
  <div class='chips'>
    <span class='chip slate'>Classical ISP stages</span>
    <span class='chip sage'>0.44 M-parameter denoiser</span>
    <span class='chip sand'>Adaptive compute per tile</span>
    <span class='chip clay'>JPG &middot; HEIC &middot; PNG &middot; DNG</span>
  </div>
</div>
"""

legend = "<div class='legend'>" + "".join(
    f"<span style='background:rgb({int(c[0] * 255)},{int(c[1] * 255)},{int(c[2] * 255)})'>{k}</span>"
    for k, c in PATH_COLORS.items()) + "</div>"
IMG_BUTTONS = ["fullscreen", "download"]
PLACEHOLDER = "<div class='placeholder'>Pick an example or upload a photo, then press <b>Process</b>.</div>"

with gr.Blocks(title="Hybrid Smartphone ISP") as demo:
    gr.HTML(HEADER)
    with gr.Row(equal_height=False):
        with gr.Column(scale=5, min_width=320):
            gr.HTML("<div class='step'><span class='n'>1</span>Choose a photo</div>")
            inp = gr.File(show_label=False, type="filepath", height=130,
                          file_types=[".dng", ".DNG", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic", ".HEIC",
                                      ".heif", ".nef", ".cr2", ".arw"])
            picker = None
            if SHOWN:
                picker = gr.Gallery([(t, c) for t, c, _ in SHOWN], label="Examples - click one to run it",
                                    columns=3, height=300, allow_preview=False, object_fit="cover")
            gr.HTML("<div class='step'><span class='n'>2</span>Settings</div>")
            noise = gr.Slider(0.0, 1.0, value=0.8, step=0.05, label="Sensor noise (low light)",
                              info="How noisy the simulated capture is (JPG / HEIC / PNG inputs)")
            budget = gr.Radio(list(BUDGET_UI), value="Balanced", label="Compute budget")
            with gr.Accordion("Advanced", open=False):
                full = gr.Checkbox(False, label=f"Full resolution (default: centre {MAX_PIXELS / 1e6:.0f} MP crop)")
            run = gr.Button("Process photo", variant="primary", size="lg")
        with gr.Column(scale=7):
            gr.HTML("<div class='step'><span class='n'>3</span>Result</div>")
            if HAS_SLIDER:
                cmp_ = gr.ImageSlider(label="Noisy capture  |  Hybrid output", type="numpy",
                                      buttons=IMG_BUTTONS, max_height=540)
            else:
                cmp_ = gr.Image(label="Hybrid output", type="numpy", format="png", buttons=IMG_BUTTONS)
            summary = gr.HTML(PLACEHOLDER)
            show_btn = gr.Button("Show full-size comparison", variant="secondary", size="md")
    shown = gr.State(False)
    with gr.Row(visible=False, elem_classes="fullrow") as full_row:
        f_noisy = gr.Image(label="Noisy capture", type="numpy", format="png", buttons=IMG_BUTTONS)
        f_trad = gr.Image(label="Traditional ISP", type="numpy", format="png", buttons=IMG_BUTTONS)
        f_hyb = gr.Image(label="Hybrid ISP (ours)", type="numpy", format="png", buttons=IMG_BUTTONS)
    with gr.Tabs(elem_classes="tabs-wrap"):
        with gr.Tab("Zoomed comparison"):
            gr.HTML("<p class='muted'>2x zoom on the noisiest region, picked automatically from the noise map.</p>")
            with gr.Row():
                z_noisy = gr.Image(label="Noisy capture", type="numpy", format="png", buttons=IMG_BUTTONS)
                z_trad = gr.Image(label="Traditional ISP", type="numpy", format="png", buttons=IMG_BUTTONS)
                z_hyb = gr.Image(label="Hybrid ISP (ours)", type="numpy", format="png", buttons=IMG_BUTTONS)
        with gr.Tab("Compute route"):
            gr.HTML(legend + "<p class='muted'>Each tile gets the cheapest path that is good enough: flat, clean "
                    "tiles use the classical filter; noisy or detailed tiles use the learned models.</p>")
            route_img = gr.Image(show_label=False, type="numpy", format="png", buttons=IMG_BUTTONS)
        with gr.Tab("ISP stages"):
            gallery = gr.Gallery(show_label=False, columns=4, height="auto", format="png")
        with gr.Tab("Details"):
            details = gr.Markdown()

    outputs = [cmp_, summary, z_noisy, z_trad, z_hyb, route_img, gallery, details, f_noisy, f_trad, f_hyb]

    def toggle(is_shown):
        is_shown = not is_shown
        label = "Hide full-size comparison" if is_shown else "Show full-size comparison"
        return gr.update(visible=is_shown), gr.update(value=label), is_shown

    show_btn.click(toggle, shown, [full_row, show_btn, shown])
    run.click(process, [inp, noise, budget, full], outputs, concurrency_limit=2)
    if picker is not None:
        def pick(evt: gr.SelectData):
            return SHOWN[evt.index][2]

        picker.select(pick, None, inp).then(process, [inp, noise, budget, full], outputs, concurrency_limit=2)

if __name__ == "__main__":
    demo.queue(max_size=16).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")),
                                   theme=THEME, css=CSS)
