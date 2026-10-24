"""Read-only preflight for a grouped MBDD2025 split and VOC annotations."""

import argparse
import collections
import json
from pathlib import Path

from PIL import Image

from mbdd_data import CLASSES, CURATED_MANIFEST_SHA256, parse_voc_boxes, read_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).resolve().parents[1] /
                        "MBDD2025_curated_v1/split_manifest.csv")
    args = parser.parse_args()
    rows, digest = read_manifest(args.manifest)
    curated_manifest = Path(__file__).resolve().parents[1] / "MBDD2025_curated_v1/split_manifest.csv"
    if args.manifest.resolve() == curated_manifest.resolve() and digest != CURATED_MANIFEST_SHA256:
        raise ValueError("Curated split manifest bytes have changed")
    image_counts = collections.Counter()
    instance_counts = {split: collections.Counter() for split in ("train", "val", "test")}
    for row in rows:
        split, image_id = row["Split"], row["ImageID"]
        image_path = args.data_root / "JPEGImages" / (image_id + ".jpg")
        annotation_path = args.data_root / "Annotations" / (image_id + ".xml")
        with Image.open(image_path) as image:
            width, height = image.size
        if (width, height) != (int(row["Width"]), int(row["Height"])):
            raise ValueError(f"Image size mismatch: {image_id}")
        _, labels = parse_voc_boxes(annotation_path, width, height)
        image_counts[split] += 1
        instance_counts[split].update(CLASSES[label - 1] for label in labels)
    print(json.dumps({"manifest_sha256": digest, "images": image_counts,
                      "instances": instance_counts}, indent=2))


if __name__ == "__main__":
    main()
