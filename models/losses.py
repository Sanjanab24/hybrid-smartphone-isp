"""Training losses: L1 + SSIM + colour (Delta E 76 in Lab) + optional LPIPS."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gauss(ws=11, sigma=1.5, ch=3, device="cpu"):
    g = torch.exp(-((torch.arange(ws, device=device) - ws // 2) ** 2).float() / (2 * sigma**2))
    g = (g / g.sum())[:, None] @ (g / g.sum())[None, :]
    return g.expand(ch, 1, ws, ws).contiguous()


def ssim_torch(x, y, ws=11):
    c = x.shape[1]
    w = _gauss(ws, ch=c, device=x.device).to(x.dtype)
    mu_x = F.conv2d(x, w, groups=c)
    mu_y = F.conv2d(y, w, groups=c)
    sxx = F.conv2d(x * x, w, groups=c) - mu_x**2
    syy = F.conv2d(y * y, w, groups=c) - mu_y**2
    sxy = F.conv2d(x * y, w, groups=c) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sxx + syy + c2))
    return s.mean()


_M = torch.tensor([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                   [0.0193339, 0.1191920, 0.9503041]])
_WHITE = torch.tensor([0.95047, 1.0, 1.08883])


def srgb_to_lab(x):
    """Differentiable sRGB (B,3,H,W) in [0,1] -> CIELAB."""
    x = x.clamp(1e-6, 1)
    lin = torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    xyz = torch.einsum("kc,bchw->bkhw", _M.to(x), lin) / _WHITE.to(x).view(1, 3, 1, 1)
    d = 6 / 29
    f = torch.where(xyz > d**3, xyz.clamp_min(1e-8) ** (1 / 3), xyz / (3 * d * d) + 4 / 29)
    L = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    return torch.stack([L, a, b], 1)


class RestorationLoss(nn.Module):
    def __init__(self, w_ssim=0.2, w_color=0.002, w_lpips=0.0):
        super().__init__()
        self.w_ssim, self.w_color, self.w_lpips = w_ssim, w_color, w_lpips
        self.lpips = None
        if w_lpips > 0:
            import lpips

            self.lpips = lpips.LPIPS(net="alex", verbose=False).eval()
            for p in self.lpips.parameters():
                p.requires_grad_(False)

    def forward(self, pred, target):
        pred32, target32 = pred.float(), target.float()
        loss = F.l1_loss(pred32, target32)
        logs = {"l1": loss.item()}
        if self.w_ssim:
            s = 1 - ssim_torch(pred32, target32)
            loss = loss + self.w_ssim * s
            logs["ssim"] = s.item()
        if self.w_color:
            de = (srgb_to_lab(pred32.clamp(0, 1)) - srgb_to_lab(target32)).pow(2).sum(1).add(1e-6).sqrt().mean()
            loss = loss + self.w_color * de
            logs["de76"] = de.item()
        if self.lpips is not None:
            self.lpips.to(pred.device)
            lp = self.lpips(pred32.clamp(0, 1) * 2 - 1, target32 * 2 - 1).mean()
            loss = loss + self.w_lpips * lp
            logs["lpips"] = lp.item()
        return loss, logs
