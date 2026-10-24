# MBDD2025 building-defect detection

This repository contains the team's reproducible **Faster R-CNN** and
**RetinaNet** baselines on a curated [MBDD2025](https://doi.org/10.5281/zenodo.15622584)
building-surface defect dataset. It includes the training and checkpointing
code, fixed split and curation audit trail, raw validation metrics, selected
predictions, summary tables, learning curves and diagnostic confusion matrices.
The image corpus is obtained separately; the trained full checkpoints are
provided as GitHub release assets rather than stored in Git history.

The five detection classes are **crack, leakage, abscission, corrosion, bulge**.
These are building-surface labels. This repository makes no bridge, severity or
natural-language-reporting claim.

## Reproduced baseline and results

| Setting | Faster R-CNN | RetinaNet |
| --- | ---: | ---: |
| Backbone | ResNet-50 FPN V2 | ResNet-50 FPN V2 |
| Initialisation | COCO detector weights | COCO detector weights |
| Images: train / validation / test | 8,418 / 1,804 / 1,804 | 8,418 / 1,804 / 1,804 |
| Input, batch, seed | 640 × 640, 1, 42 | 640 × 640, 1, 42 |
| Training | 100 epochs | 100 epochs |
| Selected validation epoch | 25 | 25 |
| Selected validation mAP50 | 0.18623 | 0.10523 |
| Selected validation mAP50–95 | **0.07613** | **0.04253** |
| Epoch-100 validation mAP50–95 | 0.06270 | 0.03681 |

Both runs used SGD (learning rate 0.0025, momentum 0.9, weight decay 0.0005),
a StepLR factor of 0.1 every 20 epochs, FP16 mixed precision, validation every
five epochs and the highest validation mAP50–95 for checkpoint selection.
Input images were aspect-preserving resized and padded. **No random data
augmentation was used in these baseline runs.** Both jobs completed 100 epochs
with exit status 0. Each ran on one **NVIDIA Quadro RTX 6000** (23.46 GiB
reported VRAM) on UTS CETUS, requesting six CPU cores, 64 GB CPU RAM and a
48-hour walltime. The saved `run.json` and PBS logs document the exact runs.
Machine-specific home paths in these copies were replaced with portable
placeholders; numeric metrics, settings and checkpoint hashes were preserved.

The values above are **validation** values as fractions, not percentages. No
held-out test metrics have been generated here. They must not be described as
test performance, and the test split must not be used for model selection.

## Figures and data

| Model | Learning curves | Confusion matrix |
| --- | --- | --- |
| Faster R-CNN | [PNG](figures/learning_curves_fasterrcnn_curated_seed42.png) | [PNG](figures/confusion_fasterrcnn_curated_seed42.png) |
| RetinaNet | [PNG](figures/learning_curves_retinanet_curated_seed42.png) | [PNG](figures/confusion_retinanet_curated_seed42.png) |

The curves combine the 100 per-epoch training losses from
`results/logs/` with mAP50 and mAP50–95 from every fifth-epoch validation JSON
under `results/<model>/`. The selected epoch-25 predictions are also included.
The confusion matrices use **validation** predictions with score ≥ 0.50 and
one-to-one, class-agnostic matching at IoU ≥ 0.50. Rows are reference classes;
columns are predicted classes; the background row contains unmatched
predictions and the background column contains missed reference boxes. These
fixed-threshold counts are diagnostics, **not** COCO mAP scores.

`tables/` contains the selected and per-epoch metrics, per-class AP, training
loss and the numerical confusion matrices. To regenerate tables and figures:

```bash
python3 -m pip install -r requirements-report.txt
python3 analysis/report_results.py --data-root MBDD2025_curated_v1
```

The dataset's `Annotations/` directory is required to recompute confusion
matrices. Without `--data-root`, the script can regenerate curves and summary
tables from the checked-in metrics and logs.

For a quick, dependency-free consistency check of the committed evidence:

```bash
python3 scripts/verify_artifacts.py
```

## Augmentation study

A controlled comparison of training with and without data augmentation, for
both detectors. All four runs used the same machine (one **NVIDIA GeForce RTX
5070 Ti Laptop GPU**, PyTorch 2.11.0, torchvision 0.26.0), the same curated
split, seed 42, 640 × 640 input, batch 1 and the baseline optimiser and
schedule, for **40 epochs**. Within each detector the two arms differ only in
`--augmentation`. The baseline arms were retrained on the laptop rather than
reusing the cluster runs above: the same Faster R-CNN baseline reached 0.164
validation mAP50 after 40 epochs on the laptop against 0.186 on CETUS, a
platform gap as large as the effect being measured.

`--augmentation standard` applies, to the training split only:

| Transform | Setting |
| --- | --- |
| Scale jitter | 0.5–1.5× of the letterbox scale, random placement and crop on the 640 × 640 canvas |
| Horizontal flip | p = 0.5 (no vertical flip: leakage stains run downwards) |
| Brightness, contrast, saturation | ±25% each |
| Hue | ±0.03 |

Boxes are transformed with the image and clipped to the canvas. A box is
dropped if a side falls below 2 px or less than 25% of its area remains. If a
draw would leave a labelled image empty, it is redrawn up to ten times, then the
unaugmented placement is used. With `--augmentation none` the dataset output is
bit-identical to the baseline code. Validation and test images are never
augmented.

Checkpoints were selected on validation mAP50–95 (every five epochs), and each
selected checkpoint was then evaluated **once** on the held-out test split:

| Run | Selected epoch | Val mAP50–95 | Test mAP50 | Test mAP50–95 |
| --- | ---: | ---: | ---: | ---: |
| Faster R-CNN, none | 30 | 0.0722 | **0.2158** | **0.0915** |
| Faster R-CNN, standard | 35 | 0.0780 | 0.2025 | 0.0824 |
| RetinaNet, none | 5 | 0.0415 | 0.1699 | 0.0665 |
| RetinaNet, standard | 40 | 0.0446 | 0.1865 | 0.0817 |

Augmentation kept the final training loss about three times higher and moved
the validation peak later in both detectors, so it delayed overfitting as
intended. Its effect on test accuracy depended on the architecture: RetinaNet
improved, Faster R-CNN regressed. Per class, bulge and abscission gained in both
detectors, while leakage (−0.109 and −0.033 AP50) and corrosion (−0.036 and
−0.022) lost, which suggests colour jitter removes a cue those two classes rely
on. With one seed and 105 test scenes, differences of about 0.01 mAP are not
separable from run-to-run variation.

Raw evidence is in `results/augmentation_study/<run>/`: `run.json`,
`train.log`, every validation metrics JSON and `test_eval/test_metrics.json`
with its log. `summary_all_runs.csv` collects the headline and per-class
numbers. Prediction JSONs (3–38 MB each) and checkpoints are not committed.
Figures: [examples](figures/augmentation_examples.png),
[validation curves](figures/augmentation_validation_curves.png),
[test comparison](figures/augmentation_test_comparison.png) and
[per-class change](figures/augmentation_per_class_delta.png).

To rerun all four arms and their test evaluations unattended on one GPU, with
the curated images placed in `MBDD2025_curated_v1/`:

```bash
python scripts/run_all.py            # resumable; safe to stop and re-run
python scripts/run_all.py --status   # progress and estimated time left
```

## Dataset curation and split

The source had **14,471 images**. The curation proposal removed **2,445**:
168 exact copies, one exact image with at most one-pixel annotation jitter,
and 2,276 near-repeated views. The proposal retained the most informative
labelled representative under its conservative rules; all removed IDs and their
representatives appear in `curation/removed_images.csv`. The curated derivative
has **12,026 images, 48,163 boxes and 2,810 scene groups**. Its split is
approximately 70/15/15 by image count and moves whole scene groups, not
individual frames. The final manifest SHA-256 is
`6f4ab573cd61f1a28120120b10126feda16635e98a0d7db22270dcaba82007c1`.

The original images were never overwritten. During copying, 99 out-of-bounds
boxes across 88 images were clipped to image bounds; YOLO labels were then
regenerated from VOC XML. No defect classes were relabelled. The curation
screening cannot rule out all semantically related scenes or mislabels. Review
the proposal, near-duplicate candidate lists and manual scene decisions under
`curation/audit_inputs/`, and the final verification under
`curation/verification/`. The exact split is `curation/split_manifest.csv`.

To reproduce the derivative from the original MBDD2025 source (which must have
`JPEGImages/` and `Annotations/`), use a **new output directory**:

```bash
python3 cetus_models/materialize_curated_dataset.py \
  --source-root /path/to/original/MBDD2025 \
  --output-root /path/to/new/MBDD2025_curated_v1 \
  --manifest curation/audit_inputs/source_split_manifest.csv \
  --proposal curation/audit_inputs/image_selection_proposal.csv \
  --pairs curation/audit_inputs/all_near_duplicate_candidates.csv \
  --wide-pairs curation/audit_inputs/all_near_duplicate_candidates_wide.csv \
  --manual-scene-links curation/audit_inputs/manual_scene_links.csv
```

## Training, checkpointing and evaluation

See [the CETUS instructions](cetus_models/README.md) for dependency setup,
upload, PBS submission and resume commands. `cetus_models/train_detector.py`
saves `last.pt` atomically every 15 minutes or 1,000 batches, at epoch end,
and on a handled termination signal. It saves `best.pt` whenever validation
mAP50–95 improves. A full checkpoint contains model, optimizer, scheduler,
loss scaler, epoch/batch progress and random-number states. Resuming uses the
same run name and rejects a changed seed, split, image size, learning rate or
batch size. A hard node failure can still lose progress since the last save.

`cetus_models/metrics.py` implements the COCO-style evaluator.
`cetus_models/evaluate_predictions.py` accepts prediction JSON with 1-based
manifest row `image_id`, class IDs 1–5 and original-image pixel `xywh` boxes;
see `cetus_models/predictions_example.json`. For example:

```bash
python3 cetus_models/evaluate_predictions.py \
  --data-root MBDD2025_curated_v1 --split val \
  --predictions results/fasterrcnn_curated_seed42/val_epoch_25_predictions.json \
  --output /tmp/fasterrcnn_val_metrics.json
```

The full trained checkpoints are described in
[checkpoints/README.md](checkpoints/README.md). The exact source and curated
manifest hashes, model settings, raw validation JSON and plotted CSVs remain
in Git so that the reported results can be audited without downloading
hundreds of megabytes of model weights.
