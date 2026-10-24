"""Read-only MBDD2025 quality audit. Never edits source images or labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from mbdd_data import CLASSES, EXPECTED_MANIFEST_SHA256, read_manifest

CLASS_TO_YOLO = {name: index for index, name in enumerate(CLASSES)}


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def xml_objects(path: Path) -> tuple[tuple[int, int] | None, list[tuple[str, tuple[float, ...]]]]:
    root = ET.parse(path).getroot()
    size = root.find("size")
    xml_size = None
    if size is not None:
        xml_size = (int(float(size.findtext("width", "nan"))),
                    int(float(size.findtext("height", "nan"))))
    objects = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        box = tuple(float(obj.findtext(f"bndbox/{key}", "nan"))
                    for key in ("xmin", "ymin", "xmax", "ymax"))
        objects.append((name, box))
    return xml_size, objects


def yolo_objects(path: Path) -> list[tuple[int, tuple[float, ...]]]:
    objects = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Expected 5 fields on line {line_number}")
        objects.append((int(parts[0]), tuple(float(x) for x in parts[1:])))
    return objects


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return overlap / (area_a + area_b - overlap) if area_a + area_b > overlap else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).resolve().parents[1] /
                        "curation/audit_inputs/source_split_manifest.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-per-class", type=int, default=20)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest, digest = read_manifest(args.manifest)
    bundled_manifest = (Path(__file__).resolve().parents[1] /
                        "curation/audit_inputs/source_split_manifest.csv").resolve()
    if args.manifest.resolve() == bundled_manifest and digest != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Bundled baseline manifest differs from Nitin's baseline")

    issues: list[dict] = []
    box_review: list[dict] = []
    images_by_hash = defaultdict(list)
    class_counts = {split: Counter() for split in ("train", "val", "test")}
    images_by_class = defaultdict(list)
    image_annotation_signatures = {}
    image_counts = Counter()
    box_counts = Counter()

    def issue(image_id, split, code, detail, severity="review"):
        issues.append({"ImageID": image_id, "Split": split,
                       "Issue": code, "Severity": severity, "Detail": detail})

    for row in manifest:
        image_id, split = row["ImageID"], row["Split"]
        width, height = int(row["Width"]), int(row["Height"])
        images_by_hash[row["SHA256"]].append(row)
        image_counts[split] += 1
        image_path = args.data_root / "JPEGImages" / f"{image_id}.jpg"
        xml_path = args.data_root / "Annotations" / f"{image_id}.xml"
        yolo_path = args.data_root / "Labels" / f"{image_id}.txt"
        for path, code in ((image_path, "missing_image"), (xml_path, "missing_xml"),
                           (yolo_path, "missing_yolo")):
            if not path.is_file():
                issue(image_id, split, code, str(path), "blocking")
        if not all(path.is_file() for path in (image_path, xml_path, yolo_path)):
            continue
        try:
            with Image.open(image_path) as image:
                actual_size = image.size
                image.verify()
            if actual_size != (width, height):
                issue(image_id, split, "image_size_mismatch",
                      f"manifest={width}x{height}; actual={actual_size}", "blocking")
        except Exception as error:
            issue(image_id, split, "image_read_error", str(error), "blocking")
            continue
        try:
            xml_size, xml = xml_objects(xml_path)
        except Exception as error:
            issue(image_id, split, "xml_parse_error", str(error), "blocking")
            continue
        if xml_size != (width, height):
            issue(image_id, split, "xml_size_mismatch",
                  f"manifest={width}x{height}; xml={xml_size}", "blocking")
        try:
            yolo = yolo_objects(yolo_path)
        except Exception as error:
            issue(image_id, split, "yolo_parse_error", str(error), "blocking")
            yolo = []
        if len(xml) != len(yolo):
            issue(image_id, split, "annotation_count_mismatch",
                  f"XML={len(xml)}; YOLO={len(yolo)}", "blocking")
        if not xml:
            issue(image_id, split, "no_objects", "Image has no labelled objects")
        image_annotation_signatures[image_id] = tuple(sorted(
            (name, *(round(value, 2) for value in box)) for name, box in xml))
        for name in {name for name, _ in xml}:
            images_by_class[name].append(image_id)
        for index, (name, box) in enumerate(xml):
            box_counts["total"] += 1
            class_counts[split][name] += 1
            if name not in CLASS_TO_YOLO:
                issue(image_id, split, "unknown_class", name, "blocking")
                continue
            if not all(math.isfinite(value) for value in box):
                issue(image_id, split, "nonfinite_box", f"object {index+1}: {box}", "blocking")
                continue
            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:
                issue(image_id, split, "degenerate_box", f"object {index+1}: {box}", "blocking")
                continue
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                box_counts["out_of_bounds"] += 1
                clipped_area = max(0, min(width, x2) - max(0, x1)) * max(0, min(height, y2) - max(0, y1))
                retention = clipped_area / ((x2-x1) * (y2-y1))
                severity = "blocking" if retention < 0.90 else "review"
                issue(image_id, split, "out_of_bounds_box",
                      f"object {index+1} {name}: {box}; retained_area={retention:.3f}", severity)
                box_review.append({"ImageID": image_id, "Split": split, "Class": name,
                                   "Reason": "out_of_bounds", "ObjectIndex": index + 1,
                                   "Value": round(retention, 4)})
            area_fraction = (x2-x1) * (y2-y1) / (width * height)
            if area_fraction < 1e-5:
                box_counts["very_small"] += 1
                box_review.append({"ImageID": image_id, "Split": split, "Class": name,
                                   "Reason": "very_small", "ObjectIndex": index + 1,
                                   "Value": round(area_fraction, 8)})
            if area_fraction > 0.50:
                box_counts["very_large"] += 1
                box_review.append({"ImageID": image_id, "Split": split, "Class": name,
                                   "Reason": "very_large", "ObjectIndex": index + 1,
                                   "Value": round(area_fraction, 4)})
            if index < len(yolo):
                yolo_class, yolo_box = yolo[index]
                expected = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                            (x2 - x1) / width, (y2 - y1) / height)
                difference = max(abs(a-b) for a, b in zip(expected, yolo_box))
                if yolo_class != CLASS_TO_YOLO[name] or difference > 1e-5:
                    box_counts["xml_yolo_mismatch"] += 1
                    issue(image_id, split, "xml_yolo_mismatch",
                          f"object {index+1} {name}: class={yolo_class}; max_norm_diff={difference:.6g}",
                          "blocking")
                if (yolo_class not in range(5) or not all(math.isfinite(v) for v in yolo_box)
                        or yolo_box[2] <= 0 or yolo_box[3] <= 0):
                    issue(image_id, split, "invalid_yolo_box",
                          f"object {index+1}: {yolo_class} {yolo_box}", "blocking")
        for left in range(len(xml)):
            for right in range(left+1, len(xml)):
                if xml[left][0] != xml[right][0] and iou(xml[left][1], xml[right][1]) >= 0.80:
                    box_counts["cross_class_overlap"] += 1
                    box_review.append({"ImageID": image_id, "Split": split,
                                       "Class": f"{xml[left][0]}+{xml[right][0]}",
                                       "Reason": "cross_class_overlap", "ObjectIndex": f"{left+1},{right+1}",
                                       "Value": round(iou(xml[left][1], xml[right][1]), 4)})

    duplicate_rows = []
    for digest_value, group in images_by_hash.items():
        if len(group) < 2:
            continue
        canonical = min(group, key=lambda row: int(row["ImageID"].replace("Hefei", "")))
        consistent = len({image_annotation_signatures.get(row["ImageID"]) for row in group}) == 1
        if not consistent:
            box_counts["duplicate_groups_with_annotation_differences"] += 1
        for row in group:
            duplicate_rows.append({"ImageID": row["ImageID"], "Split": row["Split"],
                                   "GroupID": row["GroupID"], "ImageSHA256": digest_value,
                                   "CanonicalID": canonical["ImageID"],
                                   "AnnotationExactlyEqual": consistent,
                                   "ProposedAction": "keep" if row is canonical else "review_duplicate"})

    rng = random.Random(2026)
    sample_rows = []
    for name in CLASSES:
        population = sorted(set(images_by_class[name]))
        for image_id in rng.sample(population, min(args.sample_per_class, len(population))):
            split = next(row["Split"] for row in manifest if row["ImageID"] == image_id)
            sample_rows.append({"ImageID": image_id, "Split": split, "Class": name,
                                "Reason": "stratified_visual_sample", "ObjectIndex": "", "Value": ""})

    write_csv(args.output_dir / "issues.csv", issues,
              ["ImageID", "Split", "Issue", "Severity", "Detail"])
    write_csv(args.output_dir / "exact_duplicates.csv", duplicate_rows,
              ["ImageID", "Split", "GroupID", "ImageSHA256", "CanonicalID",
               "AnnotationExactlyEqual", "ProposedAction"])
    write_csv(args.output_dir / "review_queue.csv", box_review + sample_rows,
              ["ImageID", "Split", "Class", "Reason", "ObjectIndex", "Value"])
    summary = {
        "manifest_sha256": digest,
        "images_by_split": dict(image_counts),
        "instances_by_split": {split: dict(counts) for split, counts in class_counts.items()},
        "box_counts": dict(box_counts),
        "issues_by_type": dict(Counter(row["Issue"] for row in issues)),
        "blocking_issue_images": len({row["ImageID"] for row in issues if row["Severity"] == "blocking"}),
        "exact_duplicate_groups": sum(len(group) > 1 for group in images_by_hash.values()),
        "extra_exact_duplicate_images": sum(len(group)-1 for group in images_by_hash.values()),
        "cross_split_exact_duplicate_groups": sum(
            len({row["Split"] for row in group}) > 1 for group in images_by_hash.values()),
        "review_queue_rows": len(box_review) + len(sample_rows),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
