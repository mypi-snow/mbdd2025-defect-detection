"""One COCO-style evaluator for every detector track, in original image coordinates."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch
from mbdd_data import CLASSES, parse_voc_boxes


def ground_truth(dataset) -> dict:
    images = []
    annotations = []
    annotation_id = 1
    for image_number, row in dataset.rows:
        width, height = int(row["Width"]), int(row["Height"])
        images.append({"id": image_number, "file_name": row["ImageID"] + ".jpg",
                       "width": width, "height": height})
        path = dataset.data_root / "Annotations" / (row["ImageID"] + ".xml")
        boxes, labels = parse_voc_boxes(path, width, height)
        for (x1, y1, x2, y2), label in zip(boxes, labels):
            annotations.append({"id": annotation_id, "image_id": image_number,
                                "category_id": label, "bbox": [x1, y1, x2-x1, y2-y1],
                                "area": (x2-x1) * (y2-y1), "iscrowd": 0})
            annotation_id += 1
    return {"images": images, "annotations": annotations,
            "categories": [{"id": i + 1, "name": name} for i, name in enumerate(CLASSES)],
            "info": {"description": "MBDD2025 grouped split"}, "licenses": []}


def score_predictions(gt: dict, predictions: list[dict]) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    known_images = {image["id"] for image in gt["images"]}
    for prediction in predictions:
        if prediction["image_id"] not in known_images:
            raise ValueError(f"Prediction for image outside this split: {prediction['image_id']}")
        if prediction["category_id"] not in range(1, len(CLASSES) + 1):
            raise ValueError(f"Invalid category ID: {prediction['category_id']}")
    coco_gt = COCO()
    coco_gt.dataset = gt
    coco_gt.createIndex()
    coco_dt = COCO()
    coco_dt.dataset = {
        "images": gt["images"], "categories": gt["categories"],
        "annotations": [
            {**p, "id": i + 1, "area": p["bbox"][2] * p["bbox"][3], "iscrowd": 0}
            for i, p in enumerate(predictions)
        ],
        "info": gt["info"], "licenses": [],
    }
    coco_dt.createIndex()
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = sorted(known_images)
    evaluator.params.catIds = list(range(1, len(CLASSES) + 1))
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    precision = evaluator.eval["precision"]  # IoU, recall, class, area, maxDets
    class_metrics = {}
    for index, name in enumerate(CLASSES):
        vals_50 = precision[0, :, index, 0, 2]
        vals_all = precision[:, :, index, 0, 2]
        valid_50 = vals_50[vals_50 >= 0]
        valid_all = vals_all[vals_all >= 0]
        class_metrics[name] = {
            "instances": sum(a["category_id"] == index + 1 for a in gt["annotations"]),
            "AP50": float(valid_50.mean()) if len(valid_50) else None,
            "AP50_95": float(valid_all.mean()) if len(valid_all) else None,
        }
    return {
        "mAP50": float(evaluator.stats[1]),
        "mAP50_95": float(evaluator.stats[0]),
        "AP_small": float(evaluator.stats[3]),
        "AP_medium": float(evaluator.stats[4]),
        "AP_large": float(evaluator.stats[5]),
        "per_class": class_metrics,
        "image_count": len(gt["images"]),
        "prediction_count": len(predictions),
    }


@torch.inference_mode()
def collect_predictions(model, loader, device: torch.device, dataset, score_floor: float = 0.001):
    model.eval()
    row_by_id = {image_number: row for image_number, row in dataset.rows}
    predictions = []
    timings = []
    for images, targets in loader:
        images = [image.to(device) for image in images]
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) / len(images))
        for output, target in zip(outputs, targets):
            image_number = int(target["image_id"])
            row = row_by_id[image_number]
            width, height = int(row["Width"]), int(row["Height"])
            scale = min(dataset.image_size / width, dataset.image_size / height)
            new_width, new_height = round(width * scale), round(height * scale)
            left = (dataset.image_size - new_width) // 2
            top = (dataset.image_size - new_height) // 2
            for box, label, score in zip(output["boxes"].cpu().tolist(),
                                          output["labels"].cpu().tolist(),
                                          output["scores"].cpu().tolist()):
                if score < score_floor or label not in range(1, len(CLASSES) + 1):
                    continue
                x1 = max(0.0, min(float(width), (box[0] - left) / scale))
                y1 = max(0.0, min(float(height), (box[1] - top) / scale))
                x2 = max(0.0, min(float(width), (box[2] - left) / scale))
                y2 = max(0.0, min(float(height), (box[3] - top) / scale))
                if x2 <= x1 or y2 <= y1:
                    continue
                predictions.append({"image_id": image_number, "category_id": int(label),
                                    "bbox": [x1, y1, x2-x1, y2-y1], "score": float(score)})
    return predictions, {"mean_seconds_per_image": statistics.mean(timings),
                         "median_seconds_per_image": statistics.median(timings),
                         "sequential_fps": 1 / statistics.mean(timings)}


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
