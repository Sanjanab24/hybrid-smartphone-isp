import numpy as np
import pytest
from skimage import data

from isp.bayer import flip_rggb, mosaic_rggb, pack_rggb, to_rggb, unpack_rggb
from isp.demosaic import demosaic
from isp.metrics import psnr
from isp.noise import add_raw_noise, estimate_nlf
from isp.pipeline import RenderConfig, render, traditional_isp
from isp.raw_io import RawImage, from_srgb
from isp.router import BUDGETS, run_routed


def _img():
    return data.astronaut()[:256, :256].astype(np.float32) / 255


def test_pack_roundtrip():
    x = np.random.rand(8, 10).astype(np.float32)
    assert np.array_equal(unpack_rggb(pack_rggb(x)), x)


@pytest.mark.parametrize("pattern,offset", [("RGGB", (0, 0)), ("GRBG", (0, 1)), ("GBRG", (1, 0)), ("BGGR", (1, 1))])
def test_to_rggb(pattern, offset):
    # build a mosaic whose layout is `pattern`, check the crop starts on R
    rgb = np.zeros((9, 9, 3), np.float32)
    rgb[..., 0] = 1.0  # pure red scene: only R sites are non-zero
    full = np.zeros((10, 10, 3), np.float32)
    full[..., 0] = 1
    m = mosaic_rggb(full)  # RGGB
    dy, dx = offset
    shifted = np.pad(m, ((dy, 0), (dx, 0)))  # now starts with `pattern`
    out, off = to_rggb(shifted, pattern)
    assert off == offset
    assert out[0, 0] == 1 and out[0, 1] == 0 and out[1, 1] == 0


@pytest.mark.parametrize("method", ["bilinear", "mhc", "ea"])
def test_demosaic_quality(method):
    img = _img()
    rec = demosaic(mosaic_rggb(img), method)
    assert psnr(rec[4:-4, 4:-4], img[4:-4, 4:-4]) > (27 if method == "bilinear" else 29)


def test_opencv_pattern_matches_rggb():
    img = np.zeros((32, 32, 3), np.float32)
    img[..., 0] = 0.8
    rec = demosaic(mosaic_rggb(img), "ea")
    assert rec[8:-8, 8:-8, 0].mean() > 0.7 and rec[8:-8, 8:-8, 2].mean() < 0.1


def test_flip_keeps_rggb():
    full = np.zeros((10, 10, 3), np.float32)
    full[..., 0] = 1
    m = mosaic_rggb(full)
    for fx, fy in [(1, 0), (0, 1), (1, 1)]:
        f = flip_rggb(m, fx, fy)
        assert f[0, 0] == 1 and f[1, 1] == 0


def test_unprocess_roundtrip_noiseless():
    img = _img()
    raw = from_srgb(img, noise_level=0.0)
    raw.bayer = mosaic_rggb(np.zeros_like(img))  # placeholder, replaced below
    from isp.unprocess import unprocess

    u = unprocess(img, rng=np.random.default_rng(1), noise_level=0.0)
    clean = RawImage(u["bayer_clean"], u["wb_gains"], u["cam2srgb"], u["nlf"], "synthetic")
    out = render(clean, RenderConfig(demosaic="mhc")).srgb
    assert psnr(out[4:-4, 4:-4], img[4:-4, 4:-4]) > 28


def test_nlf_estimation():
    rng = np.random.default_rng(0)
    clean = mosaic_rggb(np.tile(np.linspace(0.05, 0.9, 512, dtype=np.float32)[None, :, None], (512, 1, 3)))
    a, b = 4e-3, 2e-5
    noisy = add_raw_noise(clean, a, b, rng)
    ea, eb = estimate_nlf(pack_rggb(noisy))
    assert 0.5 * a < ea < 1.5 * a


def test_traditional_improves_noisy():
    img = _img()
    raw = from_srgb(img, noise_level=0.7, seed=3)
    noisy = render(raw).srgb
    den, _ = traditional_isp(raw)
    assert psnr(den, img) > psnr(noisy, img) + 2


def test_noise_map_tracks_error():
    img = _img()
    raw = from_srgb(img, noise_level=0.7, seed=3)
    r = render(raw)
    err = np.std(r.srgb - img)
    assert 0.4 * err < r.noise_map.mean() < 2.5 * err


@pytest.mark.parametrize("budget", list(BUDGETS))
def test_router_identity(budget):
    """With identity restorers the blended output must equal the input."""
    img = np.random.rand(300, 420, 3).astype(np.float32)
    nmap = np.full_like(img, 0.02)
    ident = lambda x: x[:, :3]  # noqa: E731
    res = run_routed(img, nmap, {"classical": ident, "student": ident, "teacher": ident}, budget, tile=128, overlap=16)
    assert np.allclose(res.image, img, atol=1e-5)
    assert abs(sum(res.fractions.values()) - 1) < 1e-6
