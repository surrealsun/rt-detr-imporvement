# Tiny RT-DETR for Aerial Tiny-Object Detection

This repository is a compact research-style project for tiny-object detection on aerial imagery. It implements an RT-DETR-inspired detector with three targeted upgrades for AI-TOD:

1. a shallow detail enhancement branch that preserves early high-resolution cues,
2. learnable multi-scale fusion with normalized per-scale weights,
3. a training-only auxiliary dense supervision head that improves shallow gradient flow and disappears at inference.

The codebase is organized to support clean baseline comparisons, ablations, checkpointed training, COCO-style evaluation, visualization, and reproducible experiment configs.

## Paper-Style Contributions

- **Detail-preserving RT-DETR adaptation for aerial tiny objects:** early backbone features are explicitly injected back into the detection stream to recover fine contours and local texture that are often lost in deeper stages.
- **Inspectable learnable multi-scale fusion:** three aligned feature levels are fused using normalized learned weights, making the scale contribution measurable during training instead of hard-coded.
- **Zero-cost inference auxiliary supervision:** a dense auxiliary head supervises the enriched mid-resolution feature map during training, then is skipped entirely in evaluation and deployment.

## Method Overview

The detector uses a ResNet backbone, projects `C3/C4/C5` to a shared hidden dimension, and runs an RT-DETR-style top-k query selection pipeline:

- `C2` feeds the **detail enhancement branch**.
- `C3/C4/C5` feed the **feature stream**.
- When enabled, a **learnable fusion block** aligns and fuses `C3/C4/C5` at the `C4` spatial scale.
- The detail branch is added into the enriched mid-level stream to recover tiny-object structure.
- The final three-scale features are flattened and passed through a lightweight transformer encoder-decoder with encoder-driven top-k query selection.
- The main detection head predicts class logits and normalized boxes with Hungarian matching and DETR losses.
- During training only, an **auxiliary dense head** predicts class heatmaps and normalized boxes on the enriched mid-level feature map.

Total loss:

`L_total = L_cls + L_box + L_giou + lambda_aux * L_aux`

where `L_aux` is the auxiliary dense heatmap plus box regression loss.

## Repository Layout

```text
configs/
  datasets/
  experiments/
data/
engine/
models/
  backbone/
  fusion/
  heads/
  losses/
scripts/
tools/
utils/
train.py
eval.py
infer.py
ablate.py
reports/
```

## Installation

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset Setup

### AI-TOD

The primary dataset is expected in COCO-style format.

```text
data/datasets/AI-TOD/
  annotations/
    train.json
    val.json
  train/
    images/
      xxx.jpg
  val/
    images/
      yyy.jpg
```

If your AI-TOD split names differ, override them in config:

```bash
python train.py --config configs/experiments/full_aitod.yaml --set dataset.train_images=train2017 dataset.val_images=val2017
```

### VisDrone2019 Optional Support

VisDrone is supported as a secondary dataset if converted to COCO JSON. A converter is included:

```bash
python tools/convert_visdrone_to_coco.py \
  --images data/datasets/VisDrone2019/train/images \
  --annotations data/datasets/VisDrone2019/train/annotations \
  --output data/datasets/VisDrone2019/annotations/train_coco.json
```

Repeat for validation annotations and then use `configs/experiments/full_visdrone.yaml`.

## Copy-Paste Only Checklist (Windows PowerShell)

Use the following commands in order from a fresh PowerShell window.

### 1. Open the repo

```powershell
cd c:\Users\curvy\Desktop\DL-Project
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `pycocotools` fails to install on Windows, install a compatible wheel for your Python version and then run:

```powershell
pip install -r requirements.txt
```

### 3. Put AI-TOD in the expected folder layout

```text
data\datasets\AI-TOD\
  annotations\
    train.json
    val.json
  train\images\
  val\images\
```

### 4. Run a 1-epoch smoke test to confirm everything works

CPU version:

```powershell
python train.py --config configs/experiments/full_aitod.yaml --set training.epochs=1 training.batch_size=1 runtime.device=cpu runtime.amp=false model.backbone.pretrained=false project.output_dir=outputs/aitod_smoke
```

CUDA version:

```powershell
python train.py --config configs/experiments/full_aitod.yaml --set training.epochs=1 training.batch_size=1 runtime.device=cuda runtime.amp=true model.backbone.pretrained=false project.output_dir=outputs/aitod_smoke
```

### 5. Open the smoke-test outputs

Check these files after the run finishes:

- `outputs\aitod_smoke\train.log`
- `outputs\aitod_smoke\history.jsonl`
- `outputs\aitod_smoke\best.pth`
- `outputs\aitod_smoke\eval\metrics.json`
- `outputs\aitod_smoke\eval\predictions\`
- `outputs\aitod_smoke\training_curves.png`
- `outputs\aitod_smoke\fusion_weights.png`

### 6. Train the baseline model

```powershell
.\scripts\train_baseline.ps1
```

### 7. Train the full proposed model

```powershell
.\scripts\train_full.ps1
```

### 8. Evaluate the full model

```powershell
.\scripts\eval_aitod.ps1 -Checkpoint outputs\aitod_full\best.pth
```

### 9. Run inference on your own image or folder

```powershell
python infer.py --config configs/experiments/full_aitod.yaml --checkpoint outputs/aitod_full/best.pth --source path\to\image_or_folder --output-dir outputs\demo
```

Open:

- `outputs\demo\`

### 10. Run all ablations and generate the summary table

```powershell
.\scripts\run_ablation.ps1
```

Open:

- `reports\ablation_summary.md`
- `reports\ablation_summary.csv`

### 11. Resume training later if needed

```powershell
python train.py --config configs/experiments/full_aitod.yaml --resume outputs/aitod_full/last.pth
```

## Training

### Baseline RT-DETR-style model

```bash
python train.py --config configs/experiments/baseline_aitod.yaml
```

### Full proposed model

```bash
python train.py --config configs/experiments/full_aitod.yaml
```

### Resume training

```bash
python train.py --config configs/experiments/full_aitod.yaml --resume outputs/aitod_full/last.pth
```

### Useful override examples

```bash
python train.py --config configs/experiments/full_aitod.yaml --set training.batch_size=4 training.epochs=36
python train.py --config configs/experiments/full_aitod.yaml --set runtime.device=cpu runtime.amp=false
```

## Evaluation

Evaluate a trained checkpoint on AI-TOD:

```bash
python eval.py --config configs/experiments/full_aitod.yaml --checkpoint outputs/aitod_full/best.pth
```

The evaluator computes:

- `mAP@[.5:.95]`
- `AP50`
- `AP75`
- `APsmall`
- parameter count
- latency / FPS
- FLOPs when `thop` is available

Metrics are written to `eval/metrics.json`, and COCO predictions are saved to `eval/coco_predictions.json`.

## Inference and Visualization

Run inference on a single image or a folder:

```bash
python infer.py --config configs/experiments/full_aitod.yaml --checkpoint outputs/aitod_full/best.pth --source path/to/image_or_folder
```

Outputs:

- box visualizations under the requested output directory,
- evaluation visualizations for up to 10 validation images,
- training curves in `training_curves.png`,
- fusion weight traces in `fusion_weights.png`.

## Ablations

Included AI-TOD experiment configs:

- `configs/experiments/baseline_aitod.yaml`
- `configs/experiments/detail_aitod.yaml`
- `configs/experiments/fusion_aitod.yaml`
- `configs/experiments/aux_aitod.yaml`
- `configs/experiments/full_aitod.yaml`

Recommended ablation order:

1. baseline
2. baseline + detail branch
3. baseline + detail branch + learnable fusion
4. baseline + detail branch + auxiliary supervision
5. full model

Aggregate completed runs into a table:

```bash
python ablate.py --run-dirs \
  outputs/aitod_baseline \
  outputs/aitod_detail \
  outputs/aitod_detail_fusion \
  outputs/aitod_detail_aux \
  outputs/aitod_full
```

This writes:

- `reports/ablation_summary.md`
- `reports/ablation_summary.csv`

## Why This Should Help Tiny Objects

- Tiny targets rely heavily on early edge and texture signals that deeper downsampled features often suppress.
- Multi-scale contribution is not constant across aerial scenes; learnable normalized fusion lets the model adaptively emphasize the most useful scale combination.
- Dense local supervision provides direct training signal to the enriched shallow-mid feature stream, improving optimization while preserving inference speed.

## Inference-Time Auxiliary Removal

The auxiliary dense head is executed only when `model.training == True`. During `eval.py` and `infer.py`, the branch is skipped, so the final detector has no extra inference-time path from auxiliary supervision.

## Outputs and Reproducibility

Each run directory stores:

- `resolved_config.yaml`
- `train.log`
- `history.jsonl`
- `best.pth`
- `last.pth`
- `eval/metrics.json`
- `training_curves.png`
- `fusion_weights.png` when learnable fusion is enabled

Seed control is exposed via config, and deterministic behavior is enabled where PyTorch permits it.

## Assumptions and Limitations

- This repo expects COCO-style AI-TOD annotations and does not bundle dataset files.
- Distributed training is not included; the implementation is single-process and research-friendly rather than cluster-oriented.
- Benchmark gains are not claimed in advance. Run the included training and evaluation scripts to generate actual ablation numbers in your environment.
- The model is RT-DETR-inspired and keeps the key real-time query selection flavor, but it is intentionally compact to remain suitable for a mini-project.
