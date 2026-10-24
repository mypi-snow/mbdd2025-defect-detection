"""Apply the same COCO evaluator to detections exported by any model.

Input is a JSON list of {image_id, category_id, bbox, score}, with bbox in
original-image xywh pixels and IDs from split_manifest.csv (row number, 1-based).
"""

import argparse
import json
from pathlib import Path

from mbdd_data import MBDDDataset
from metrics import ground_truth, score_predictions, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).resolve().parents[1] /
                        "MBDD2025_curated_v1/split_manifest.csv")
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = MBDDDataset(args.data_root, args.manifest, args.split)
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    result = score_predictions(ground_truth(dataset), predictions)
    result["split"] = args.split
    result["manifest_sha256"] = dataset.manifest_sha256
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
