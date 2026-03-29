# Project Summary: Tiny-Object RT-DETR with Detail, Fusion, and Auxiliary Supervision

## Motivation

RT-DETR is an appealing one-stage transformer detector for efficient object detection, but aerial tiny objects are difficult because aggressive downsampling weakens high-frequency cues and because equal treatment of multi-scale features is rarely optimal across scenes. AI-TOD is especially sensitive to missing shallow structure because many targets occupy only a few pixels.

## Proposed Detector

The implemented detector keeps an RT-DETR-style backbone-to-query pipeline and adds three targeted components:

### 1. Shallow Detail Enhancement Branch

- taps early `C2` features from the ResNet backbone,
- processes them with lightweight depthwise-separable convolution blocks,
- aligns them to the mid-level `C4` scale,
- injects them into the detection stream.

This branch is intended to preserve edges, local gradients, object boundaries, and appearance micro-patterns useful for tiny-object localization.

### 2. Learnable Multi-Scale Fusion

- aligns `C3`, `C4`, and `C5` into a shared spatial resolution,
- predicts normalized scale weights using learnable logits plus an optional content-aware gate,
- fuses the scales into an enriched mid-level representation,
- logs the learned weights for interpretability.

This replaces fixed fusion heuristics with an adaptive mechanism that can shift emphasis across scenes and datasets.

### 3. Training-Only Auxiliary Dense Supervision

- applies a lightweight dense head on the enriched feature map,
- supervises local class heatmaps and normalized boxes,
- helps propagate gradients into the shallow/detail path,
- is disabled automatically in evaluation and inference.

This preserves the main model's inference-time cost while improving optimization during training.

## Loss

The training objective is:

`L_total = L_cls + L_box + L_giou + lambda_aux * L_aux`

where:

- `L_cls` is DETR-style classification loss,
- `L_box` is normalized L1 box regression loss,
- `L_giou` is generalized IoU loss,
- `L_aux` is the dense auxiliary supervision loss.

## Evaluation Protocol

The codebase is set up to report:

- `mAP@[.5:.95]`
- `AP50`
- `APsmall`
- parameter count
- FLOPs when profiling is available
- latency / FPS

The ablation configs isolate the contribution of:

1. the baseline detector,
2. the shallow detail branch,
3. learnable fusion,
4. auxiliary supervision,
5. the full model.

## Expected Outcome

The intended effect is strongest improvement on `APsmall` and overall `mAP@[.5:.95]`, especially on AI-TOD where tiny-object evidence is sparse and easily lost. No benchmark numbers are claimed here; the repository is designed to generate them reproducibly through actual runs.
