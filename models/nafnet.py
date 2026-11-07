"""NAFNet (Chen et al., ECCV 2022) with options needed by the hybrid ISP.

Module/parameter names match the official megvii-research/NAFNet code, so the
released SIDD checkpoints load directly. Extensions:
  * in_ch != out_ch: extra input channels (noise map) with residual on the image channels
  * use_norm=False: norm-free blocks (cheaper on CPU, friendlier to INT8)
  * out_shuffle: PixelShuffle output head (packed RAW at H/2 -> RGB at H)
Padding to a multiple of 2**levels is done by the caller (export friendly).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / torch.sqrt(var + self.eps)
        return self.weight.view(1, -1, 1, 1) * y + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2, use_norm: bool = True):
        super().__init__()
        dw = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw, 1, bias=True)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw, bias=True)
        self.conv3 = nn.Conv2d(dw // 2, c, 1, bias=True)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw // 2, dw // 2, 1, bias=True))
        self.sg = SimpleGate()
        ffn = ffn_expand * c
        self.conv4 = nn.Conv2d(c, ffn, 1, bias=True)
        self.conv5 = nn.Conv2d(ffn // 2, c, 1, bias=True)
        self.norm1 = LayerNorm2d(c) if use_norm else nn.Identity()
        self.norm2 = LayerNorm2d(c) if use_norm else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.sg(self.conv2(self.conv1(x)))
        x = x * self.sca(x)
        y = inp + self.conv3(x) * self.beta
        x = self.conv5(self.sg(self.conv4(self.norm2(y))))
        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, width=32, middle_blk_num=12, enc_blk_nums=(2, 2, 4, 8),
                 dec_blk_nums=(2, 2, 2, 2), use_norm=True, residual=True, out_shuffle: int = 1):
        super().__init__()
        self.in_ch, self.out_ch, self.residual, self.out_shuffle = in_ch, out_ch, residual, out_shuffle
        self.intro = nn.Conv2d(in_ch, width, 3, padding=1, bias=True)
        self.ending = nn.Conv2d(width, out_ch * out_shuffle**2, 3, padding=1, bias=True)
        self.encoders, self.decoders, self.ups, self.downs = (nn.ModuleList() for _ in range(4))
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan, use_norm=use_norm) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2
        self.middle_blks = nn.Sequential(*[NAFBlock(chan, use_norm=use_norm) for _ in range(middle_blk_num)])
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan, use_norm=use_norm) for _ in range(num)]))
        self.multiple = 2 ** len(self.encoders)

    def forward(self, inp):
        x = self.intro(inp)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for dec, up, skip in zip(self.decoders, self.ups, skips[::-1]):
            x = dec(up(x) + skip)
        x = self.ending(x)
        if self.out_shuffle > 1:
            x = F.pixel_shuffle(x, self.out_shuffle)
        if self.residual:
            x = x + inp[:, : self.out_ch]
        return x


def pad_to_multiple(x: torch.Tensor, m: int):
    h, w = x.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, (h, w)


def run_padded(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    xp, (h, w) = pad_to_multiple(x, model.multiple)
    s = model.out_shuffle
    return model(xp)[..., : h * s, : w * s]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
