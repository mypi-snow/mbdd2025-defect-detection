"""Summarise saved validation runs and plot their learning curves and box errors.

The confusion matrices are diagnostic. They use a fixed confidence threshold
and class-agnostic, one-to-one IoU matching; they are not COCO AP scores.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CLASSES = ("crack", "leakage", "abscission", "corrosion", "bulge")
MODELS = ("fasterrcnn_curated_seed42", "retinanet_curated_seed42")
DISPLAY_NAMES = {"fasterrcnn_curated_seed42": "Faster R-CNN",
                 "retinanet_curated_seed42": "RetinaNet"}
LOG_FILES = {"fasterrcnn_curated_seed42": "mbdd-detector.o157824",
             "retinanet_curated_seed42": "mbdd-detector.o157826"}
LOSS_PATTERN = re.compile(r"^epoch=(\d+) mean_loss=([0-9.]+) seconds=([0-9.]+)$")
EPOCH_PATTERN = re.compile(r"val_epoch_(\d+)_metrics\.json$")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_run(package: Path, model: str) -> tuple[dict, list[tuple[int, dict]], list[dict]]:
    """Read every saved validation point and each epoch's logged mean loss.

    We select checkpoints from validation JSON, not from the final epoch or
    from a test result. Requiring all loss rows prevents a partial log from
    producing a misleading learning curve.
    """
    directory = package / "results" / model
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    points = []
    for path in directory.glob("val_epoch_*_metrics.json"):
        match = EPOCH_PATTERN.fullmatch(path.name)
        if match:
            value = json.loads(path.read_text(encoding="utf-8"))
            if value["split"] != "val" or value["manifest_sha256"] != run["manifest_sha256"]:
                raise ValueError(f"Mismatched validation result: {path}")
            points.append((int(match.group(1)), value))
    points.sort(key=lambda item: item[0])
    if not points:
        raise ValueError(f"No validation results in {directory}")
    losses = []
    log_path = package / "results" / "logs" / LOG_FILES[model]
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = LOSS_PATTERN.fullmatch(line)
        if match:
            losses.append({"model": model, "epoch": int(match.group(1)),
                           "mean_training_loss": float(match.group(2)),
                           "training_seconds": float(match.group(3))})
    if len(losses) != run["epochs"] or [row["epoch"] for row in losses] != list(range(1, run["epochs"] + 1)):
        raise ValueError(f"Training loss log is incomplete for {model}")
    return run, points, losses


def box_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def ground_truth(package: Path, data_root: Path, expected_sha: str) -> dict[int, list[tuple[int, tuple[float, ...]]]]:
    manifest = package / "curation" / "split_manifest.csv"
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != expected_sha:
        raise ValueError("Curated split manifest does not match the training run")
    by_image = {}
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        for number, row in enumerate(csv.DictReader(stream), 1):
            if row["Split"] != "val":
                continue
            path = data_root / "Annotations" / (row["ImageID"] + ".xml")
            labels = []
            for obj in ET.parse(path).getroot().findall("object"):
                name = (obj.findtext("name") or "").strip().lower()
                if name not in CLASSES:
                    raise ValueError(f"Unexpected category {name!r} in {path}")
                bounds = obj.find("bndbox")
                if bounds is None:
                    raise ValueError(f"Missing bounding box in {path}")
                box = tuple(float(bounds.findtext(key, "nan")) for key in ("xmin", "ymin", "xmax", "ymax"))
                labels.append((CLASSES.index(name), box))
            by_image[number] = labels
    return by_image


def confusion_matrix(predictions: list[dict], truth: dict[int, list[tuple[int, tuple[float, ...]]]],
                     score_threshold: float, iou_threshold: float) -> np.ndarray:
    """Count class matches, missed references and unmatched predictions.

    Predictions are processed by descending confidence and matched to the
    highest-IoU still-unmatched reference irrespective of class. This exposes
    both localisation failures and cross-class errors at one fixed threshold;
    it is deliberately distinct from the COCO AP calculation.
    """
    background = len(CLASSES)
    matrix = np.zeros((background + 1, background + 1), dtype=np.int64)
    by_image = defaultdict(list)
    for pred in predictions:
        if pred["score"] >= score_threshold:
            image_id, category_id = int(pred["image_id"]), int(pred["category_id"])
            if image_id not in truth or not 1 <= category_id <= len(CLASSES):
                raise ValueError("Prediction does not belong to validation split or class list")
            x, y, w, h = pred["bbox"]
            by_image[image_id].append((category_id - 1, (x, y, x + w, y + h), float(pred["score"])))
    for image_id, references in truth.items():
        unmatched = set(range(len(references)))
        for predicted_class, box, _score in sorted(by_image[image_id], key=lambda item: -item[2]):
            candidate = max(unmatched, key=lambda index: box_iou(box, references[index][1]), default=None)
            if candidate is not None and box_iou(box, references[candidate][1]) >= iou_threshold:
                true_class = references[candidate][0]
                matrix[true_class, predicted_class] += 1
                unmatched.remove(candidate)
            else:
                matrix[background, predicted_class] += 1
        for index in unmatched:
            matrix[references[index][0], background] += 1
    return matrix


def plot_curves(package: Path, model: str, points: list[tuple[int, dict]],
                losses: list[dict], selected_epoch: int) -> None:
    epochs = [epoch for epoch, _ in points]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    axes[0].plot([row["epoch"] for row in losses],
                 [row["mean_training_loss"] for row in losses], color="#165a72", linewidth=1.8)
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Mean loss")
    axes[1].plot(epochs, [value["mAP50"] for _, value in points],
                 marker="o", markersize=3, color="#c67713", label="mAP50")
    axes[1].plot(epochs, [value["mAP50_95"] for _, value in points],
                 marker="o", markersize=3, color="#165a72", label="mAP50–95")
    axes[1].axvline(selected_epoch, color="#666666", linestyle="--", linewidth=1,
                    label=f"Selected epoch {selected_epoch}")
    axes[1].set(title="Validation average precision", xlabel="Epoch", ylabel="Average precision (0–1)")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=0.22)
        ax.set_xlim(left=1, right=100)
        ax.set_ylim(bottom=0)
    fig.suptitle(DISPLAY_NAMES[model] + "  |  MBDD2025 curated validation")
    fig.savefig(package / "figures" / f"learning_curves_{model}.png", dpi=180)
    plt.close(fig)


def plot_confusion(package: Path, model: str, matrix: np.ndarray,
                   score_threshold: float, iou_threshold: float) -> None:
    labels = list(CLASSES) + ["background"]
    fig, ax = plt.subplots(figsize=(9, 7.2))
    fig.subplots_adjust(left=0.18, right=0.98, top=0.88, bottom=0.24)
    colors = np.log1p(matrix.astype(float))
    ax.imshow(colors, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Reference class")
    ax.set_title(f"{DISPLAY_NAMES[model]}  |  validation box confusion\n"
                 f"Score ≥ {score_threshold:.2f}, IoU ≥ {iou_threshold:.2f}; colour uses log(count + 1)")
    for row in range(len(labels)):
        for col in range(len(labels)):
            if row == len(CLASSES) and col == len(CLASSES):
                continue
            value = int(matrix[row, col])
            ax.text(col, row, f"{value:,}", ha="center", va="center", fontsize=8,
                    color="white" if colors[row, col] > colors.max() * 0.55 else "#17212b")
    ax.axhline(len(CLASSES) - 0.5, color="#5e6b77", linewidth=0.9)
    ax.axvline(len(CLASSES) - 0.5, color="#5e6b77", linewidth=0.9)
    fig.text(0.5, 0.045,
             "Background row: unmatched predictions (false positives).\n"
             "Background column: missed references (false negatives).",
             ha="center", fontsize=9)
    fig.savefig(package / "figures" / f"confusion_{model}.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-root", type=Path, help="Extracted MBDD2025_curated_v1 folder; required to recompute confusion matrices")
    parser.add_argument("--score", type=float, default=0.50)
    parser.add_argument("--iou", type=float, default=0.50)
    args = parser.parse_args()
    if not 0 <= args.score <= 1 or not 0 < args.iou <= 1:
        parser.error("Score and IoU thresholds must lie in [0,1] and (0,1], respectively")
    package = args.package_root.resolve()
    (package / "figures").mkdir(exist_ok=True)
    (package / "tables").mkdir(exist_ok=True)
    selected_rows, all_rows, per_class_rows, loss_rows = [], [], [], []
    for model in MODELS:
        run, points, losses = load_run(package, model)
        selected_epoch, selected = max(points, key=lambda item: item[1]["mAP50_95"])
        final_epoch, final = points[-1]
        selected_rows.append({"model": model, "selected_epoch": selected_epoch,
                              "selected_val_mAP50_95": selected["mAP50_95"],
                              "selected_val_mAP50": selected["mAP50"],
                              "final_epoch": final_epoch,
                              "final_val_mAP50_95": final["mAP50_95"],
                              "training_images": run["train_images"],
                              "validation_images": run["validation_images"],
                              "gpu": run["gpu"], "gpu_memory_gb": run["gpu_memory_gb"],
                              "manifest_sha256": run["manifest_sha256"]})
        for epoch, result in points:
            all_rows.append({"model": model, "epoch": epoch, "val_mAP50": result["mAP50"],
                             "val_mAP50_95": result["mAP50_95"]})
        for name in CLASSES:
            value = selected["per_class"][name]
            per_class_rows.append({"model": model, "selected_epoch": selected_epoch,
                                   "class": name, "reference_boxes": value["instances"],
                                   "val_AP50": value["AP50"], "val_AP50_95": value["AP50_95"]})
        loss_rows.extend(losses)
        plot_curves(package, model, points, losses, selected_epoch)
        if args.data_root:
            truth = ground_truth(package, args.data_root, run["manifest_sha256"])
            predictions_path = package / "results" / model / f"val_epoch_{selected_epoch}_predictions.json"
            predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
            matrix = confusion_matrix(predictions, truth, args.score, args.iou)
            write_csv(package / "tables" / f"confusion_{model}.csv",
                      [{"reference_class": name, **dict(zip(list(CLASSES) + ["background"], row.tolist()))}
                       for name, row in zip(list(CLASSES) + ["background"], matrix)],
                      ["reference_class"] + list(CLASSES) + ["background"])
            plot_confusion(package, model, matrix, args.score, args.iou)
            print(f"{model}: selected epoch {selected_epoch}; confusion matrix computed on {len(truth)} validation images")
        else:
            print(f"{model}: selected epoch {selected_epoch}; pass --data-root to recompute confusion matrices")
    write_csv(package / "tables" / "selected_validation_metrics.csv", selected_rows,
              list(selected_rows[0]))
    write_csv(package / "tables" / "validation_by_epoch.csv", all_rows, list(all_rows[0]))
    write_csv(package / "tables" / "per_class_selected.csv", per_class_rows,
              list(per_class_rows[0]))
    write_csv(package / "tables" / "training_loss_by_epoch.csv", loss_rows,
              list(loss_rows[0]))


if __name__ == "__main__":
    main()
