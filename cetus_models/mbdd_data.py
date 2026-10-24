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
from torchvision.transforms.functional import (
    adjust_brightness,
    adjust_contrast,
    adjust_hue,
    adjust_saturation,
    pil_to_tensor,
)

CLASSES = ("crack", "leakage", "abscission", "corrosion", "bulge")
CLASS_TO_ID = {name: i + 1 for i, name in enumerate(CLASSES)}
EXPECTED_MANIFEST_SHA256 = "1ad8ec9c77ae84e64531490df1843e075ed0de2bd891f9e847e93821e9482742"
# The published detector runs used this separately curated, scene-grouped split.
CURATED_MANIFEST_SHA256 = "6f4ab573cd61f1a28120120b10126feda16635e98a0d7db22270dcaba82007c1"

AUGMENTATION_CHOICES = ("none", "standard")

# Scale jitter combined with random placement on the letterbox canvas. A factor
# below 1.0 shrinks the image and pads, above 1.0 enlarges it so the canvas
# crops it, so this single step covers scale, translation and random cropping.
SCALE_JITTER = (0.5, 1.5)
FLIP_PROBABILITY = 0.5
# Vertical flips are deliberately excluded: leakage staining runs downward
# under gravity, so a vertically flipped leakage box is not a real defect.
BRIGHTNESS_RANGE = 0.25
CONTRAST_RANGE = 0.25
SATURATION_RANGE = 0.25
HUE_RANGE = 0.03
# A transformed box is kept only if it stays this visible after cropping.
MIN_BOX_SIDE = 2.0
MIN_VISIBLE_FRACTION = 0.25
# Retry a placement that crops every box away before falling back to the
# unaugmented letterbox, so a bad draw never produces an empty training target.
AUGMENT_ATTEMPTS = 10


def _uniform(low: float, high: float) -> float:
    """Sample from the torch RNG, which DataLoader seeds per worker and epoch."""
    return low + (high - low) * float(torch.rand(()))


def photometric_jitter(tensor: torch.Tensor) -> torch.Tensor:
    tensor = adjust_brightness(tensor, _uniform(1 - BRIGHTNESS_RANGE, 1 + BRIGHTNESS_RANGE))
    tensor = adjust_contrast(tensor, _uniform(1 - CONTRAST_RANGE, 1 + CONTRAST_RANGE))
    tensor = adjust_saturation(tensor, _uniform(1 - SATURATION_RANGE, 1 + SATURATION_RANGE))
    tensor = adjust_hue(tensor, _uniform(-HUE_RANGE, HUE_RANGE))
    return tensor.clamp_(0.0, 1.0)


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
                 image_size: int = 640, limit: int | None = None,
                 augment: str = "none"):
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
        if augment not in AUGMENTATION_CHOICES:
            raise ValueError(f"Unknown augmentation {augment!r}")
        if augment != "none" and split != "train":
            raise ValueError("Augmentation is for the training split only")
        self.augment = augment
        if not self.rows:
            raise ValueError(f"No images in split {split}")

    def __len__(self) -> int:
        return len(self.rows)

    def _letterbox(self, source: Image.Image, boxes: list[list[float]],
                   labels: list[int], scale_factor: float, random_offset: bool,
                   flip: bool) -> tuple[Image.Image, list[list[float]], list[int]]:
        """Resize onto the padded square canvas and carry the boxes with it.

        scale_factor 1.0 with random_offset False reproduces the baseline
        centred letterbox exactly, so an unaugmented run is unchanged.
        """
        size = self.image_size
        width, height = source.size
        scale = min(size / width, size / height) * scale_factor
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        resized = source.resize((new_width, new_height), Image.Resampling.BILINEAR)
        if random_offset:
            # A negative offset crops the enlarged image; PIL clips the paste.
            left = int(torch.randint(min(0, size - new_width),
                                     max(0, size - new_width) + 1, ()))
            top = int(torch.randint(min(0, size - new_height),
                                    max(0, size - new_height) + 1, ()))
        else:
            left = (size - new_width) // 2
            top = (size - new_height) // 2
        canvas = Image.new("RGB", (size, size), (114, 114, 114))
        canvas.paste(resized, (left, top))
        kept_boxes: list[list[float]] = []
        kept_labels: list[int] = []
        for (x1, y1, x2, y2), label in zip(boxes, labels):
            bx1, by1 = x1 * scale + left, y1 * scale + top
            bx2, by2 = x2 * scale + left, y2 * scale + top
            full_area = max((bx2 - bx1) * (by2 - by1), 1e-6)
            cx1, cy1 = min(max(bx1, 0.0), size), min(max(by1, 0.0), size)
            cx2, cy2 = min(max(bx2, 0.0), size), min(max(by2, 0.0), size)
            if cx2 - cx1 < MIN_BOX_SIDE or cy2 - cy1 < MIN_BOX_SIDE:
                continue
            if (cx2 - cx1) * (cy2 - cy1) / full_area < MIN_VISIBLE_FRACTION:
                continue
            kept_boxes.append([cx1, cy1, cx2, cy2])
            kept_labels.append(label)
        if flip:
            canvas = canvas.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            kept_boxes = [[size - x2, y1, size - x1, y2]
                          for x1, y1, x2, y2 in kept_boxes]
        return canvas, kept_boxes, kept_labels

    def __getitem__(self, index: int):
        image_number, row = self.rows[index]
        image_id = row["ImageID"]
        image_path = self.data_root / "JPEGImages" / f"{image_id}.jpg"
        annotation_path = self.data_root / "Annotations" / f"{image_id}.xml"
        placed = None
        with Image.open(image_path) as source:
            source = source.convert("RGB")
            width, height = source.size
            if (width, height) != (int(row["Width"]), int(row["Height"])):
                raise ValueError(f"Image size differs from manifest: {image_id}")
            boxes, labels = parse_voc_boxes(annotation_path, width, height)
            if self.augment == "standard":
                for _ in range(AUGMENT_ATTEMPTS):
                    candidate = self._letterbox(
                        source, boxes, labels, _uniform(*SCALE_JITTER),
                        True, float(torch.rand(())) < FLIP_PROBABILITY)
                    if candidate[2] or not labels:
                        placed = candidate
                        break
            if placed is None:
                placed = self._letterbox(source, boxes, labels, 1.0, False, False)
        canvas, scaled_boxes, kept_labels = placed
        tensor = pil_to_tensor(canvas).float().div_(255.0)
        if self.augment == "standard":
            tensor = photometric_jitter(tensor)
        box_tensor = torch.tensor(scaled_boxes, dtype=torch.float32).reshape(-1, 4)
        label_tensor = torch.tensor(kept_labels, dtype=torch.int64)
        area = ((box_tensor[:, 2] - box_tensor[:, 0]) *
                (box_tensor[:, 3] - box_tensor[:, 1]))
        target = {
            "boxes": box_tensor,
            "labels": label_tensor,
            "image_id": torch.tensor(image_number, dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros(len(kept_labels), dtype=torch.int64),
        }
        return tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))
