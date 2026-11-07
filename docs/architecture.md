# System architecture

Every box names the module or function that implements it.

## Inference (what the app runs)

```mermaid
flowchart LR
    subgraph IN["Input - isp/raw_io.py: load_any"]
        DNG["Phone DNG / RAW<br/>load_dng (rawpy)"]
        SIDD["SIDD .MAT + metadata<br/>load_sidd"]
        JPG["JPG / PNG / HEIC<br/>from_srgb -> isp/unprocess.py<br/>(simulated noisy RAW)"]
    end

    subgraph CLS["Classical front-end - isp/pipeline.py: render"]
        BL["Black level, normalise,<br/>CFA -> RGGB<br/>isp/bayer.py"]
        WB["White balance<br/>as-shot neutral / gray-world<br/>isp/color.py"]
        DM["Demosaic (MHC / edge-aware)<br/>isp/demosaic.py"]
        CCM["Colour correction matrix<br/>camera -> XYZ -> sRGB"]
        GM["Gamma / tone<br/>(+ auto exposure)"]
        NM["Noise map: var = a*x + b<br/>propagated through WB, CCM, gamma<br/>isp/noise.py"]
    end

    subgraph RT["Adaptive compute - isp/router.py: run_routed"]
        SC["Tile difficulty<br/>noise x detail"]
        C0["Classical path<br/>guided filter, isp/filters.py"]
        S0["Student 0.44 M<br/>ONNX (models/zoo.py: student)"]
        T0["Teacher 116 M<br/>NAFNet-SIDD fine-tuned<br/>(evaluation / GPU)"]
        BLD["Feathered tile blending"]
    end

    OUT["sRGB output<br/>+ stages, route map, metrics<br/>app/app.py"]

    DNG & SIDD & JPG --> BL --> WB --> DM --> CCM --> GM
    BL --> NM
    CCM --> NM
    GM --> SC
    NM --> SC
    SC -->|easy tiles| C0
    SC -->|medium| S0
    SC -->|hard| T0
    C0 & S0 & T0 --> BLD --> OUT
```

`isp/hybrid.py: HybridISP` wires the classical front-end and the router together. The deployed app uses the
classical and student paths. The teacher path is used in evaluation and wherever a teacher ONNX is placed in `weights/`.

## Training and optimisation (offline, `scripts/`)

```mermaid
flowchart LR
    SIDDRAW["SIDD RAW + sRGB"] --> CHK["check_rendering.py<br/>per-camera CFA / CCM / demosaic"]
    SIDDRAW --> PREP["prepare_sidd.py<br/>leave-one-camera-out crops"]
    PREP --> CLEAN["make_clean_crops.py<br/>clean GT crops -> simulated RAW"]
    NAF["Pretrained NAFNet-SIDD<br/>(official weights)"] -->|"zoo.py: load_pretrained_body<br/>noise channels zero-init"| TFT["train.py --model teacher<br/>fp32, then real + simulated mix"]
    PREP --> TFT
    CLEAN --> TFT
    TFT --> KD["train.py --model student<br/>--teacher ... --distill 1.0"]
    CLEAN --> KD
    KD --> EXP["export_onnx.py<br/>stability gate + torch/ONNX check"]
    EXP --> Q["quantize.py<br/>static INT8, QDQ, per-channel"]
    NAF --> B["train.py --model e2e<br/>(learned-only baseline)"]
    EXP --> APP["app/app.py<br/>(student_fp32.onnx)"]
    TFT & KD & B & Q --> EV["evaluate.py / robustness.py / benchmark.py"] --> MD["make_eval_md.py<br/>eval_P2_T57.md"]
```

## Compared variants

| Id | Pipeline | Where |
|---|---|---|
| A | Traditional ISP: our rendering + guided-filter denoise + adaptive sharpening | `isp/pipeline.py: traditional_isp` |
| A2 | Classical denoise on the same input as the hybrid (fair classical baseline) | `isp/router.py` budget `classical_only` |
| B | Learned-only ISP: packed RAW -> sRGB end-to-end | `models/zoo.py: build("e2e")` |
| D0 | Rendering + pretrained NAFNet, no fine-tuning | `--pretrained` in `scripts/evaluate.py` |
| D | Hybrid: rendering + fine-tuned noise-aware teacher | `build("teacher")` |
| E | Hybrid with the distilled student (FP32 / INT8) | `build("student")`, `scripts/quantize.py` |
| F | Adaptive compute: per-tile routing over classical / student / teacher | `isp/router.py` |
