"""Download a pretrained NAFNet-SIDD checkpoint and check it loads into our teacher.

The official links are in the "Results and Pre-trained Models" table of
https://github.com/megvii-research/NAFNet (NAFNet-SIDD-width64 / width32, Google Drive or Baidu).

  python -m scripts.fetch_weights --url "<google-drive link from the NAFNet README>" --out weights/NAFNet-SIDD-width64.pth --width 64
  python -m scripts.fetch_weights --check weights/NAFNet-SIDD-width64.pth --width 64     # verify an existing file
"""
from __future__ import annotations

import argparse
import os
import urllib.request

from models.zoo import build


def download(url: str, out: str):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if "drive.google.com" in url:
        import gdown

        gdown.download(url, out, quiet=False, fuzzy=True)
    else:
        urllib.request.urlretrieve(url, out)


def check(path: str, width: int):
    m = build("teacher", width, path)
    print(f"OK: {path} loads into the teacher (width {width}); noise-map channels zero-initialised.")
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url")
    p.add_argument("--out", default="weights/NAFNet-SIDD-width64.pth")
    p.add_argument("--check", default=None)
    p.add_argument("--width", type=int, default=64)
    a = p.parse_args()
    if a.url:
        download(a.url, a.out)
        check(a.out, a.width)
    elif a.check:
        check(a.check, a.width)
    else:
        p.error("give --url or --check")


if __name__ == "__main__":
    main()
