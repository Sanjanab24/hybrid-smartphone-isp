# Design document - P2: Hybrid Smartphone ISP

**Team T57 - Bit Wizards** · Live app: https://sanjanab24--hybrid-isp-web.modal.run

> Each `_Write: ..._` line is a prompt. Replace it with 2-5 sentences in your own words, then delete the prompt.

## 1. Problem and goal
Smartphone RAW images contain sensor noise, Bayer mosaicing, colour errors and illumination variations. Our goal was to convert degraded RAW data into clean sRGB images while maintaining colour accuracy and detail under limited computation. We designed a hybrid ISP combining physically interpretable classical stages with learned restoration.

## 2. Architecture
Diagram: [architecture.md](architecture.md)

Pipeline: RAW -> black level / CFA -> white balance -> demosaic -> CCM -> gamma -> noise map -> learned restoration
(tile router: classical / student / teacher) -> sRGB

The classical ISP handles physically defined operations such as white balance, demosaicing, colour correction and gamma conversion. Its output and estimated noise map are then passed to the learned restoration model for adaptive denoising and detail recovery. We chose a hybrid approach because it preserves accurate, interpretable colour processing while using deep learning only where it provides the greatest improvement, resulting in better quality with lower computational cost.

## 3. Data
| Item | Value |
|---|---|
| Dataset | SIDD subset (Hugging Face mirror), 20 image pairs, 5 phones |
| Train / val / test | 14 images (GP, IP, N6, S6) / 2 images (IP) / 4 images (G4, held-out camera) |
| Crops | 48 x 512 px per train image; test = 16 crops of 1024 px |
| Simulated RAW | 560 clean crops, unprocessed into noisy RAW (Brooks et al. 2019), mixed 50/50 |
| Unseen photos | 5 test images in `eval_images/`, 3 noise levels |

We split the dataset by camera and reserved G4 for testing to measure how well the system generalizes to an unseen sensor. Because the real training set is small, we added simulated RAW images with varied noise and illumination levels to increase diversity, reduce overfitting and improve robustness under challenging conditions.

## 4. Classical ISP
| Stage | Method | File |
|---|---|---|
| Demosaic | Malvar-He-Cutler / edge-aware (chosen per camera) | `isp/demosaic.py` |
| White balance | As-shot neutral from metadata; gray-world fallback | `isp/color.py` |
| Colour correction | DNG ColorMatrix -> XYZ -> sRGB | `isp/color.py` |
| Gamma / tone | sRGB gamma, auto exposure | `isp/color.py` |
| Noise model | var = a*x + b, propagated to sRGB as a noise map | `isp/noise.py` |
| Classical denoise | Guided filter (beat bilateral and NLM by 2-3 dB) | `isp/filters.py` |
| Sharpening | Adaptive unsharp mask (weaker at high noise) | `isp/filters.py` |

White balance corrects colour casts using camera metadata, while demosaicing reconstructs a full RGB image from the Bayer sensor pattern. We also estimate a noise map that represents the noise level in each image region. This map helps the learned model apply stronger restoration to noisy areas while preserving fine details in cleaner regions.

## 5. Learned models
| Model | Params | Start point | Training |
|---|---|---|---|
| Teacher (NAFNet-SIDD w64 + noise-map input) | 116.0 M | Pretrained NAFNet-SIDD | fp32, lr 2e-5, 1500 it; then 1000 it on real + simulated mix |
| Student (3-level NAFNet, width 16) | 0.44 M | Scratch | Distilled from teacher, 6000 it; then 1500 it on the mix |
| Learned-only ISP (baseline) | 29.2 M | NAFNet-SIDD w32 body | RAW -> sRGB end-to-end, 3000 it |

Loss: L1 + 0.2 (1 - SSIM) + 0.002 Delta E76, in fp32.

We fine-tuned a pretrained NAFNet because our real training dataset was too small to train a large model reliably from scratch. Knowledge distillation transfers the teacher model’s restoration behaviour to a much smaller student, allowing it to preserve most of the image quality while requiring fewer parameters, less memory and lower inference time.

## 6. Optimisation
| Technique | Result |
|---|---|
| Knowledge distillation | 116 M -> 0.44 M params (265x smaller), -0.63 dB |
| INT8 (ONNX Runtime, static) | 1.87 MB -> 0.90 MB, 17% faster on CPU, -1.44 dB |
| FP16 on GPU | Teacher 755 -> 451 ms per MP |
| Adaptive compute (router) | "Fast" budget about 40% faster than student-only |

We deployed the FP32 student because it provides the best balance between restoration quality, model size and inference speed. It is only 1.87 MB and retains near-teacher image quality, while the INT8 version is smaller but causes a more noticeable reduction in PSNR and colour accuracy.

## 7. Results (held-out camera G4)
| Pipeline | PSNR | SSIM | Delta E | LPIPS |
|---|---|---|---|---|
| Learned-only ISP | 33.27 | 0.969 | 3.88 | 0.102 |
| Classical ISP | 41.60 | 0.967 | 1.68 | 0.120 |
| Pretrained NAFNet (no fine-tuning) | 45.70 | 0.989 | 0.82 | 0.023 |
| **Hybrid teacher** | **46.39** | **0.990** | **0.74** | **0.022** |
| **Hybrid student (deployed)** | **45.76** | 0.988 | 0.90 | 0.035 |
| Hybrid student INT8 | 44.32 | 0.978 | 1.16 | 0.030 |

| Efficiency (1 MP) | Value |
|---|---|
| Student, GPU FP16 | 39 ms |
| Full app pipeline, 2 CPU threads | about 2 s |
| Live app (Modal, 2 MP photo) | 6.8 s |

Full tables: [eval_P2_T57.md](../eval_P2_T57.md) · Figure: [sidd_comparison.jpg](figures/sidd_comparison.jpg)

The hybrid teacher achieved the best quality at 46.39 dB PSNR, while the compact student retained 45.76 dB with approximately 265 times fewer parameters. Knowledge distillation improved the student by 0.76 dB, although part of this gain may come from the additional simulated training data. The deployed app runs the 1.87 MB FP32 student on CPU and processes a 2 MP photo in about 6.8 s; on a GPU with FP16 the student needs only about 39 ms per megapixel.
## 8. Ablations and failure cases
See [ablation_and_failures.md](ablation_and_failures.md).

## 9. Deployment
| Item | Value |
|---|---|
| URL | https://sanjanab24--hybrid-isp-web.modal.run |
| Host | Modal, 2 CPUs, 4 GB RAM (`scripts/deploy_modal.py`); `Dockerfile` for any Docker host |
| Inputs | JPG, PNG, TIFF, HEIC, Bayer DNG/RAW |
| Runtime | ONNX Runtime (no PyTorch) |

## 10. Limitations
only 14 training images and 2 for validation
the pretrained teacher already saw SIDD scenes
the student smooths dense fine texture
the simulated noise is only Gaussian

## 11. Ethics
SIDD is MIT-licensed
the demo photos were taken by your team
uploads aren’t stored
outputs aren’t forensic evidence

## 12. Team
| Member | USN | Owns |
|---|---|---|
| Sanjana B (lead) | 1RUA24CSE0408 | Models, training, integration |
| Saanvi Mahipati | 1RUA24CSE0389 | Classical ISP, evaluation |
| Sanjana Gothe | 1RUA24CSE0410 | Optimisation, app, deployment |

## 13. AI assistance declaration
We used an AI assistant (Claude, by Anthropic) for system-design discussion, code implementation and debugging,
deployment setup.
All documentations , data preparation, training and evaluation runs, demo photos, design decisions and the final report text are our
team's own work, and every number in this report comes from our own runs.

## 14. References
- Chen et al., NAFNet, ECCV 2022
- Abdelhamed et al., SIDD, CVPR 2018
- Brooks et al., Unprocessing Images for Learned Raw Denoising, CVPR 2019
- Malvar, He & Cutler, demosaicing, ICASSP 2004
- He et al., Guided Image Filtering, TPAMI 2013
- Hinton et al., Distilling the Knowledge in a Neural Network, 2015
