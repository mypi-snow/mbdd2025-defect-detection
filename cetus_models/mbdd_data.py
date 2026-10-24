"""MBDD2025 dataset adapter for versioned, grouped split manifests."""

from __future__ import annotations

import csv
import hashlib
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

CLASSES = ("crack", "leakage", "abscission", "corrosion", "bulge")
CLASS_TO_ID = {name: i + 1 for i, name in enumerate(CLASSES)}
EXPECTED_MANIFEST_SHA256 = "1ad8ec9c77ae84e64531490df1843e075ed0de2bd891f9e847e93821e9482742"
# The published detector runs used this separately curated, scene-grouped split.
CURATED_MANIFEST_SHA256 = "6f4ab573cd61f1a28120120b10126feda16635e98a0d7db22270dcaba82007c1"


def read_manifest(path: str | Path) -> tuple[list[dict[str, str]], str]:
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("The split manifest is empty")
    required = {"ImageID", "Width", "Height", "Split", "GroupID"}
    if not required.issubset(rows[0]):
        raise ValueError(f"Manifest missing columns: {sorted(required - set(rows[0]))}")
    ids = [row["ImageID"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("The manifest contains duplicate ImageID values")
    groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split = row["Split"].lower()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unexpected split: {split}")
        groups[row["GroupID"]].add(split)
    leaking = [group for group, splits in groups.items() if len(splits) > 1]
    if leaking:
        raise ValueError(f"Group IDs cross splits: {leaking[:5]}")
    return rows, digest


def parse_voc_boxes(path: Path, width: int, height: int) -> tuple[list[list[float]], list[int]]:
    root = ET.parse(path).getroot()
    boxes: list[list[float]] = []
    labels: list[int] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name not in CLASS_TO_ID:
            raise ValueError(f"Unknown class {name!r} in {path}")
        bnd = obj.find("bndbox")
        if bnd is None:
            raise ValueError(f"Missing bndbox in {path}")
        values = [float(bnd.findtext(key, "nan")) for key in ("xmin", "ymin", "xmax", "ymax")]
        # The supplied YOLO labels are generated directly from these XML edge
        # coordinates, and many XML boxes start at 0. Treat them as zero-based
        # continuous edges; clipping below handles source out-of-bounds boxes.
        x1, y1, x2, y2 = values
        x1, x2 = max(0.0, x1), min(float(width), x2)
        y1, y2 = max(0.0, y1), min(float(height), y2)
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"Invalid VOC box {values} in {path}")
        boxes.append([x1, y1, x2, y2])
        labels.append(CLASS_TO_ID[name])
    return boxes, labels


class MBDDDataset(Dataset):
    def __init__(self, data_root: str | Path, manifest: str | Path, split: str,
                 image_size: int = 640, limit: int | None = None):
        self.data_root = Path(data_root)
        self.rows_all, self.manifest_sha256 = read_manifest(manifest)
        self.rows = [(i + 1, row) for i, row in enumerate(self.rows_all)
                     if row["Split"].lower() == split]
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be positive")
            self.rows = self.rows[:limit]
        self.split = split
        self.image_size = image_size
        if not self.rows:
            raise ValueError(f"No images in split {split}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        image_number, row = self.rows[index]
        image_id = row["ImageID"]
        image_path = self.data_root / "JPEGImages" / f"{image_id}.jpg"
        annotation_path = self.data_root / "Annotations" / f"{image_id}.xml"
        with Image.open(image_path) as source:
            source = source.convert("RGB")
            width, height = source.size
            if (width, height) != (int(row["Width"]), int(row["Height"])):
                raise ValueError(f"Image size differs from manifest: {image_id}")
            boxes, labels = parse_voc_boxes(annotation_path, width, height)
            scale = min(self.image_size / width, self.image_size / height)
            new_width = round(width * scale)
            new_height = round(height * scale)
            resized = source.resize((new_width, new_height), Image.Resampling.BILINEAR)
        left = (self.image_size - new_width) // 2
        top = (self.image_size - new_height) // 2
        canvas = Image.new("RGB", (self.image_size, self.image_size), (114, 114, 114))
        canvas.paste(resized, (left, top))
        tensor = pil_to_tensor(canvas).float().div_(255.0)
        scaled_boxes = [[x1 * scale + left, y1 * scale + top,
                         x2 * scale + left, y2 * scale + top]
                        for x1, y1, x2, y2 in boxes]
        box_tensor = torch.tensor(scaled_boxes, dtype=torch.float32).reshape(-1, 4)
        label_tensor = torch.tensor(labels, dtype=torch.int64)
        area = ((box_tensor[:, 2] - box_tensor[:, 0]) *
                (box_tensor[:, 3] - box_tensor[:, 1]))
        target = {
            "boxes": box_tensor,
            "labels": label_tensor,
            "image_id": torch.tensor(image_number, dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros(len(labels), dtype=torch.int64),
        }
        return tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))
