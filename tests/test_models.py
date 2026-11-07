import os

import cv2
import numpy as np
import pytest
import torch

from isp.raw_io import InputError, load_any
from models.nafnet import NAFNet, run_padded
from models.zoo import SIDD_BLOCKS, build, load_ckpt, save_ckpt


def test_pretrained_expansion_equals_original(tmp_path):
    """A 6-channel teacher initialised from a 3-channel checkpoint must equal it, whatever the noise map."""
    torch.manual_seed(0)
    orig = NAFNet(in_ch=3, out_ch=3, width=8, **SIDD_BLOCKS).eval()
    for p in orig.parameters():  # make residual branches active (official init has zero beta/gamma)
        p.data.normal_(0, 0.05)
    path = tmp_path / "naf.pth"
    torch.save({"params": orig.state_dict()}, path)
    teacher = build("teacher", width=8, pretrained=str(path), verbose=False).eval()
    rgb = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        ref = run_padded(orig, rgb)
        out1 = run_padded(teacher, torch.cat([rgb, torch.zeros_like(rgb)], 1))
        out2 = run_padded(teacher, torch.cat([rgb, torch.rand_like(rgb)], 1))
    assert torch.allclose(ref, out1, atol=1e-5)
    assert torch.allclose(ref, out2, atol=1e-5)


@pytest.mark.parametrize("name,shape_in,shape_out", [
    ("student", (1, 6, 70, 90), (1, 3, 70, 90)),
    ("e2e", (1, 4, 40, 50), (1, 3, 80, 100)),
    ("raw_denoiser", (1, 8, 40, 50), (1, 4, 40, 50)),
])
def test_shapes_and_ckpt_roundtrip(tmp_path, name, shape_in, shape_out):
    m = build(name, width=8, verbose=False).eval()
    x = torch.rand(*shape_in)
    with torch.no_grad():
        y = run_padded(m, x)
    assert tuple(y.shape) == shape_out
    p = tmp_path / "m.pt"
    save_ckpt(m, str(p), name, 8)
    m2 = load_ckpt(str(p))
    with torch.no_grad():
        assert torch.allclose(run_padded(m2, x), y)


def test_input_errors(tmp_path):
    with pytest.raises(InputError):
        load_any(str(tmp_path / "missing.jpg"))
    bad = tmp_path / "notes.txt"
    bad.write_text("hello")
    with pytest.raises(InputError):
        load_any(str(bad))
    corrupt = tmp_path / "broken.jpg"
    corrupt.write_bytes(b"\xff\xd8 not really a jpeg")
    with pytest.raises(InputError):
        load_any(str(corrupt))
    tiny = tmp_path / "tiny.png"
    cv2.imwrite(str(tiny), np.zeros((20, 20, 3), np.uint8))
    with pytest.raises(InputError):
        load_any(str(tiny))
    fake_dng = tmp_path / "fake.dng"
    fake_dng.write_bytes(os.urandom(2048))
    with pytest.raises(InputError):
        load_any(str(fake_dng))
