#!/usr/bin/env python3
"""Check that the committed split, raw metrics, tables and figures agree.

This is intentionally independent of PyTorch and the image dataset. It checks
the shareable evidence; recomputing confusion matrices or COCO AP additionally
requires the VOC annotations.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SHA256 = "6f4ab573cd61f1a28120120b10126feda16635e98a0d7db22270dcaba82007c1"
MODELS = ("fasterrcnn_curated_seed42", "retinanet_curated_seed42")
CLASSES = ("crack", "leakage", "abscission", "corrosion", "bulge")
EXPECTED = {
    "fasterrcnn_curated_seed42": (0.18623292055116444, 0.07613313696732486),
    "retinanet_curated_seed42": (0.10522781542509478, 0.04252803193484227),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def check_manifest() -> None:
    manifest = ROOT / "curation/split_manifest.csv"
    alternate = ROOT / "MBDD2025_curated_v1/split_manifest.csv"
    raw = manifest.read_bytes()
    assert alternate.read_bytes() == raw, "Dataset placement manifest differs from curation manifest"
    assert hashlib.sha256(raw).hexdigest() == MANIFEST_SHA256, "Unexpected curated manifest bytes"
    rows = read_csv(manifest)
    assert len(rows) == 12026
    assert len({row["ImageID"] for row in rows}) == len(rows), "Duplicate ImageID"
    assert Counter(row["Split"] for row in rows) == {"train": 8418, "val": 1804, "test": 1804}
    groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        groups[row["GroupID"]].add(row["Split"])
    assert len(groups) == 2810
    assert all(len(splits) == 1 for splits in groups.values()), "Scene group crosses splits"


def check_run(model: str) -> None:
    directory = ROOT / "results" / model
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    assert run["manifest_sha256"] == MANIFEST_SHA256
    assert (run["epochs"], run["image_size"], run["batch_size"], run["seed"]) == (100, 640, 1, 42)
    assert (run["train_images"], run["validation_images"]) == (8418, 1804)
    assert run["gpu"] == "Quadro RTX 6000"
    points = []
    for epoch in range(5, 101, 5):
        path = directory / f"val_epoch_{epoch}_metrics.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        assert result["split"] == "val" and result["manifest_sha256"] == MANIFEST_SHA256
        points.append((epoch, result))
    selected_epoch, selected = max(points, key=lambda pair: pair[1]["mAP50_95"])
    assert selected_epoch == 25
    expected_map50, expected_map = EXPECTED[model]
    assert abs(selected["mAP50"] - expected_map50) < 1e-12
    assert abs(selected["mAP50_95"] - expected_map) < 1e-12
    assert (directory / "val_epoch_25_predictions.json").stat().st_size > 0

    # A reference box appears once in exactly one true-class row of the
    # confusion matrix, even if it was missed (background column).
    matrix = {row["reference_class"]: row for row in
              read_csv(ROOT / "tables" / f"confusion_{model}.csv")}
    for category in CLASSES:
        total = sum(int(matrix[category][column]) for column in (*CLASSES, "background"))
        assert total == selected["per_class"][category]["instances"], (model, category, total)
    for kind in ("learning_curves", "confusion"):
        png = ROOT / "figures" / f"{kind}_{model}.png"
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", png


def main() -> None:
    check_manifest()
    for model in MODELS:
        check_run(model)
    selected = read_csv(ROOT / "tables/selected_validation_metrics.csv")
    assert {row["model"] for row in selected} == set(MODELS)
    loss = read_csv(ROOT / "tables/training_loss_by_epoch.csv")
    assert Counter(row["model"] for row in loss) == {model: 100 for model in MODELS}
    print("Verified curated split, 40 validation metrics, selected results, 200 loss rows and four figures")


if __name__ == "__main__":
    main()
