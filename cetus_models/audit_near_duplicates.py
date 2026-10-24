"""Flag visual near-duplicates for manual review, across or within splits."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from mbdd_data import EXPECTED_MANIFEST_SHA256, read_manifest


def hashes(image_path: Path) -> tuple[int, np.ndarray]:
    with Image.open(image_path) as image:
        gray = image.convert("L")
        small = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
        thumbnail = np.asarray(gray.resize((32, 18), Image.Resampling.LANCZOS), dtype=np.uint8)
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    dhash = sum(int(bit) << index for index, bit in enumerate(bits))
    return dhash, thumbnail


def band_keys(value: int):
    # Any two 64-bit hashes with <=4 differing bits share one of five bands.
    shift = 0
    for band, width in enumerate((13, 13, 13, 13, 12)):
        yield band, (value >> shift) & ((1 << width) - 1)
        shift += width


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).resolve().parents[1] /
                        "curation/audit_inputs/source_split_manifest.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-hamming", type=int, default=4)
    parser.add_argument("--max-thumbnail-difference", type=float, default=0.06)
    parser.add_argument("--include-same-split", action="store_true",
                        help="Include within-split pairs when building a curation proposal")
    args = parser.parse_args()
    if args.max_hamming > 4:
        parser.error("The five-band index guarantees candidates only through Hamming distance 4")
    rows, digest = read_manifest(args.manifest)
    bundled_manifest = (Path(__file__).resolve().parents[1] /
                        "curation/audit_inputs/source_split_manifest.csv").resolve()
    if args.manifest.resolve() == bundled_manifest and digest != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Bundled baseline manifest differs from Nitin's baseline")
    buckets = defaultdict(list)
    seen = []
    matches = []
    candidate_pairs = 0
    for index, row in enumerate(rows):
        image_id = row["ImageID"]
        value, thumbnail = hashes(args.data_root / "JPEGImages" / (image_id + ".jpg"))
        candidates = set()
        for key in band_keys(value):
            candidates.update(buckets[key])
        for previous in candidates:
            previous_row, previous_hash, previous_thumbnail = seen[previous]
            if not args.include_same_split and previous_row["Split"] == row["Split"]:
                continue
            candidate_pairs += 1
            hamming = (value ^ previous_hash).bit_count()
            if hamming > args.max_hamming:
                continue
            difference = float(np.abs(thumbnail.astype(np.int16) -
                                      previous_thumbnail.astype(np.int16)).mean() / 255.0)
            if difference <= args.max_thumbnail_difference:
                matches.append({"ImageID_A": previous_row["ImageID"],
                                "Split_A": previous_row["Split"],
                                "GroupID_A": previous_row["GroupID"],
                                "ImageID_B": image_id, "Split_B": row["Split"],
                                "GroupID_B": row["GroupID"],
                                "DHashDistance": hamming,
                                "ThumbnailDifference": round(difference, 6)})
        seen.append((row, value, thumbnail))
        for key in band_keys(value):
            buckets[key].append(index)
        if (index + 1) % 3000 == 0:
            print(f"Scanned {index+1}/{len(rows)} images; flagged {len(matches)} pairs", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["ImageID_A", "Split_A", "GroupID_A",
                                                       "ImageID_B", "Split_B", "GroupID_B",
                                                       "DHashDistance", "ThumbnailDifference"])
        writer.writeheader()
        writer.writerows(sorted(matches, key=lambda row: (row["ThumbnailDifference"],
                                                        row["DHashDistance"])))
    scope = "all" if args.include_same_split else "cross_split"
    print(json.dumps({"scanned_images": len(rows), "scope": scope,
                      "candidate_pairs": candidate_pairs,
                      "flagged_pairs": len(matches), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
