# Ablations and failure cases

The numbers come from `results/*.json`. Held-out camera G4 (16 crops) unless stated otherwise.

## Ablations
| Change | Variant | PSNR | Delta E | Effect |
|---|---|---|---|---|
| Full hybrid (reference) | teacher | 46.39 | 0.74 | - |
| No fine-tuning | pretrained NAFNet | 45.70 | 0.82 | fine-tuning +0.69 dB |
| Learned colour instead of classical | learned-only ISP | 33.27 | 3.88 | classical WB/CCM +13.1 dB |
| No learned restoration | classical only | 41.60 | 1.68 | learned restoration +4.8 dB |
| Teacher -> student (265x smaller) | student FP32 | 45.76 | 0.90 | -0.63 dB |
| No distillation | student, GT only | 45.00 | 0.98 | distillation +0.76 dB (\*) |
| FP32 -> INT8 | student INT8 | 44.32 | 1.16 | -1.44 dB |
| Router -> random routing | balanced budget | 44.03 vs 43.79 | 1.21 vs 1.23 | router +0.24 dB |

(\*) The no-distillation student was trained on SIDD only, without the simulated-RAW mix, so part of this gain may
come from the extra data. Not run: a separately trained teacher without the noise map.

The biggest single gain comes from keeping colour processing classical. The learned-only ISP is 13.1 dB worse because
it has to learn white balance and colour correction from only 14 images, while the hybrid takes them from camera
metadata. On top of the classical pipeline, learned restoration adds 4.8 dB over the best classical denoiser, and
fine-tuning the pretrained teacher adds another 0.69 dB.

Compression is cheap: the student is 265x smaller than the teacher but only 0.63 dB worse, and distillation helps it
by up to 0.76 dB. INT8 costs more (1.44 dB). The router beats random tile assignment by only 0.24 dB.

## Robustness (PSNR, dB)
| Condition | Traditional | Hybrid teacher | Student INT8 |
|---|---|---|---|
| Noise x1 | 42.27 | 49.07 | 45.47 |
| Noise x2 | 39.49 | 47.59 | 44.49 |
| Noise x4 | 36.58 | 45.09 | 43.08 |
| Noise x8 | 33.65 | 41.51 | 41.33 |
| Under-exposed x0.5 | 38.22 | 46.17 | 43.71 |
| Under-exposed x0.25 | 33.73 | 41.39 | 40.68 |

Plot: [robustness.png](figures/robustness.png)

The hybrid stays 7-8 dB ahead of the traditional ISP at every noise level and exposure. At 8x noise the 0.44 M
student almost matches the 116 M teacher (41.33 vs 41.51 dB), so the gap between the models shrinks as the input gets harder.

## Failure cases
**1. Dense texture in downscaled images** (`eval_images/`): classical 31.11 dB / LPIPS 0.179 vs student 27.83 dB /
LPIPS 0.331; the student's Delta E is still lower (2.95 vs 3.37).
* **Cause:** the student was trained on full-resolution sensor images, where detail is spread over many pixels.
  Downscaled images pack fine texture into single pixels, so the small student treats it as noise and smooths it
  away. Photos straight from a phone camera do not show this (e.g. `IMG_1030.HEIC`: +6.2 dB over classical).
* **Fix:** train on natural high-resolution photos (e.g. DIV2K), or give the student more capacity.

**2. Wrong white-balance metadata:** Delta E 5.55 vs 0.75 with correct WB; the gray-world fallback is worse (Delta E 10.62).
* **Cause:** colour correction is classical by design, so the learned stage does not fix a wrong white balance coming
  from the metadata. Gray-world assumes the scene averages to grey, which is false for colourful scenes.
* **Fix:** replace the gray-world fallback with a learned illuminant estimator.

**3. INT8 quantisation:** -1.44 dB and +0.25 Delta E versus FP32, for only a 17% CPU speed-up, so the deployed app uses FP32.
* **Cause:** the outputs of the LayerNorm and gating layers have wide value ranges, and 8-bit quantisation loses
  precision on them.
* **Fix:** quantisation-aware training, or keeping these layers in FP16 while the rest runs in INT8.

**4. Router vs random on unseen photos:** balanced router 28.51 dB vs random routing 29.07 dB.
* **Cause:** the difficulty score sends highly textured tiles to the student, but those are exactly the tiles where
  the student over-smooths (failure case 1).
* **What it means:** the router is mainly a speed control (the "Fast" budget is about 40% faster than student-only),
  not a quality improvement. Its difficulty score would need to depend on the image content to also help quality.

**Training issues we hit (fixed):**
- bf16 loss computation diverged (37.3 -> 22 dB); losses now run in fp32.
- A norm-free student produced NaN outputs on extreme inputs; LayerNorm restored and an export stability gate added.
