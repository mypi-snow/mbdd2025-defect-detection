"""Download verified torchvision COCO weights once, then copy them to CETUS."""

import argparse
import os
from pathlib import Path

import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    RetinaNet_ResNet50_FPN_V2_Weights,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["fasterrcnn", "retinanet"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    weights = (FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if args.model == "fasterrcnn"
               else RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / f"{args.model}_coco_v2.pt"
    temp = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    with temp.open("wb") as stream:
        torch.save(weights.get_state_dict(progress=True, check_hash=True), stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, destination)
    print(destination)


if __name__ == "__main__":
    main()
