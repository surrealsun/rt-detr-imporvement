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

### PennFudanPed Tiny Public Fallback

If you do not want to download or construct AI-TOD yet, this repo now supports a very small public dataset based on **PennFudanPed**.

What this gives you:

- automatic download,
- automatic train/val split,
- automatic COCO-format conversion,
- tiny enough for CPU-laptop demos,
- a clean way to prove the project runs end to end.

This is **not** an aerial tiny-object benchmark, so use it for functional validation and demo outputs, not for claiming aerial tiny-object gains.

Download and prepare it automatically:

```powershell
.\scripts\download_pennfudan.ps1
```

This stores the dataset under:

```text
data\datasets\PennFudanCOCO\
  annotations\
    train.json
    val.json
  train\images\
  val\images\
  raw\
```

The downloader uses the public PennFudanPed zip from the official Penn dataset page:

- `https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip`

You can also run the downloader directly:

```powershell
python tools\download_pennfudan.py --output-root data\datasets\PennFudanCOCO
```

Train the full demo model on PennFudan:

```powershell
.\scripts\train_pennfudan_cpu_demo.ps1
```

Train the baseline demo model on PennFudan:

```powershell
.\scripts\train_baseline_pennfudan_cpu_demo.ps1
```

Configs:

- `configs/experiments/cpu_demo_pennfudan.yaml`
- `configs/experiments/baseline_cpu_demo_pennfudan.yaml`

### TinyPennFudan Synthetic Tiny-Object Fallback

PennFudan remains available exactly as before. In addition, the repo now includes a second public fallback called **TinyPennFudan**, generated automatically from PennFudan.

What TinyPennFudan does:

- keeps PennFudan as the public source dataset,
- builds larger synthetic canvases,
- pastes strongly downscaled PennFudan mini-scenes onto those canvases,
- turns the pedestrians into much smaller objects,
- gives you a more tiny-object-like fallback than plain PennFudan.

Important:

- this is still a synthetic fallback, not a real aerial dataset,
- it is meant to stress tiny-object behavior more than standard PennFudan,
- use it as a stronger laptop demo, not as a substitute for AI-TOD benchmark claims.

Build the TinyPennFudan dataset:

```powershell
.\scripts\build_tinypennfudan.ps1
```

This creates:

```text
data\datasets\TinyPennFudanCOCO\
  annotations\
    train.json
    val.json
  train\images\
  val\images\
```

Train the baseline tiny-object fallback model:

```powershell
.\scripts\train_baseline_tinypennfudan_cpu_demo.ps1
```

Train the full proposed tiny-object fallback model:

```powershell
.\scripts\train_tinypennfudan_cpu_demo.ps1
```

Configs:

- `configs/experiments/baseline_cpu_demo_tinypennfudan.yaml`
- `configs/experiments/cpu_demo_tinypennfudan.yaml`

## Small AI-TOD Subset for Laptop Use

If you only have a CPU laptop with limited disk and RAM, the most realistic path is to work with a **small local subset** of AI-TOD.

Important note:

- As of March 29, 2026, the official AI-TOD release does **not** provide a clean official "mini subset download" package.
- The official instructions still say you need the xView training set plus `AI-TOD_wo_xview` and the synthesis toolkit to construct AI-TOD.

Practical options:

1. ask a teammate or lab machine for an already-generated AI-TOD copy, then extract a small subset locally,
2. build the full AI-TOD once on another machine and copy only a small subset to your laptop,
3. if you already have full AI-TOD on disk, use the included subset tool below.

### Create a small subset from a full AI-TOD copy

This script copies only a small number of train/val images and writes matching COCO annotations:

- `tools/make_coco_subset.py`

Example:

```powershell
python tools\make_coco_subset.py --source-root data\datasets\AI-TOD --output-root data\datasets\AI-TOD-mini --train-count 200 --val-count 50
```

This creates:

```text
data\datasets\AI-TOD-mini\
  annotations\
    train.json
    val.json
  train\images\
  val\images\
```

Train on the subset without changing files by overriding the dataset root:

```powershell
python train.py --config configs/experiments/cpu_demo_aitod.yaml --set dataset.root=data/datasets/AI-TOD-mini
```

### Very small subset example

For a really lightweight laptop demo:

```powershell
python tools\make_coco_subset.py --source-root data\datasets\AI-TOD --output-root data\datasets\AI-TOD-tiny --train-count 50 --val-count 20
python train.py --config configs/experiments/cpu_demo_aitod.yaml --set dataset.root=data/datasets/AI-TOD-tiny dataset.train_max_samples=50 dataset.val_max_samples=20 training.epochs=1
```

### Recommended presentation path on a laptop

If your goal is to show that the project works and compare variants, use:

1. `AI-TOD-mini` with 50 to 200 training images,
2. `baseline_cpu_demo_aitod.yaml`,
3. `cpu_demo_aitod.yaml`,
4. the generated logs, metrics, prediction images, and curves.

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

## Easiest No-AI-TOD Demo Path

If you just want this repository to run on a normal laptop, use PennFudan first:

```powershell
cd c:\Users\curvy\Desktop\DL-Project
.\.venv\Scripts\Activate.ps1
.\scripts\download_pennfudan.ps1
.\scripts\train_baseline_pennfudan_cpu_demo.ps1
.\scripts\train_pennfudan_cpu_demo.ps1
python ablate.py --run-dirs outputs/pennfudan_baseline_cpu_demo outputs/pennfudan_cpu_demo --output reports/pennfudan_demo_summary.md --csv reports/pennfudan_demo_summary.csv
```

Then open:

- `outputs\pennfudan_baseline_cpu_demo\eval\metrics.json`
- `outputs\pennfudan_cpu_demo\eval\metrics.json`
- `outputs\pennfudan_baseline_cpu_demo\eval\predictions\`
- `outputs\pennfudan_cpu_demo\eval\predictions\`
- `reports\pennfudan_demo_summary.md`

## Tiny-Object-Like Laptop Demo Path

If you want a fallback that is more aligned with tiny-object behavior than plain PennFudan, use TinyPennFudan:

```powershell
cd c:\Users\curvy\Desktop\DL-Project
.\.venv\Scripts\Activate.ps1
.\scripts\build_tinypennfudan.ps1
.\scripts\train_baseline_tinypennfudan_cpu_demo.ps1
.\scripts\train_tinypennfudan_cpu_demo.ps1
python ablate.py --run-dirs outputs/tinypennfudan_baseline_cpu_demo outputs/tinypennfudan_cpu_demo --output reports/tinypennfudan_demo_summary.md --csv reports/tinypennfudan_demo_summary.csv
```

Then open:

- `outputs\tinypennfudan_baseline_cpu_demo\eval\metrics.json`
- `outputs\tinypennfudan_cpu_demo\eval\metrics.json`
- `outputs\tinypennfudan_baseline_cpu_demo\eval\predictions\`
- `outputs\tinypennfudan_cpu_demo\eval\predictions\`
- `reports\tinypennfudan_demo_summary.md`

## Low-Resource Laptop Mode

If you have a CPU-only laptop with around 16 GB RAM, do **not** try to run the full default AI-TOD setup first. Use the included low-resource demo config instead.

What this mode changes:

- uses `cpu` only,
- uses `workers: 0`,
- reduces image scale to `384`,
- switches to `resnet18`,
- reduces transformer size and number of queries,
- trains on a small subset of AI-TOD by default,
- still produces logs, metrics, prediction images, curves, and fusion-weight plots.

Included config:

- `configs/experiments/cpu_demo_aitod.yaml`
- `configs/experiments/baseline_cpu_demo_aitod.yaml`

Included script:

- `scripts/train_cpu_demo.ps1`
- `scripts/train_baseline_cpu_demo.ps1`

Run it:

```powershell
.\scripts\train_cpu_demo.ps1
```

Or directly:

```powershell
python train.py --config configs/experiments/cpu_demo_aitod.yaml
```

Baseline CPU demo:

```powershell
.\scripts\train_baseline_cpu_demo.ps1
```

Or directly:

```powershell
python train.py --config configs/experiments/baseline_cpu_demo_aitod.yaml
```

Default subset in low-resource mode:

- `train_max_samples: 200`
- `val_max_samples: 50`
- `epochs: 2`

Outputs will be written to:

- `outputs\aitod_cpu_demo\`

Open these after the run:

- `outputs\aitod_cpu_demo\train.log`
- `outputs\aitod_cpu_demo\history.jsonl`
- `outputs\aitod_cpu_demo\eval\metrics.json`
- `outputs\aitod_cpu_demo\eval\predictions\`
- `outputs\aitod_cpu_demo\training_curves.png`
- `outputs\aitod_cpu_demo\fusion_weights.png`

If it is still too slow, reduce the subset even more:

```powershell
python train.py --config configs/experiments/cpu_demo_aitod.yaml --set dataset.train_max_samples=50 dataset.val_max_samples=20 training.epochs=1
```

If you want a slightly stronger run and your laptop can handle it:

```powershell
python train.py --config configs/experiments/cpu_demo_aitod.yaml --set dataset.train_max_samples=500 dataset.val_max_samples=100 training.epochs=3
```

This mode is best for:

- proving the code runs end to end,
- generating screenshots and example outputs,
- showing ablation wiring and experiment structure,
- demonstrating the proposed method against a baseline on a manageable subset.

This mode is **not** intended to produce paper-quality final benchmark numbers on AI-TOD.

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
