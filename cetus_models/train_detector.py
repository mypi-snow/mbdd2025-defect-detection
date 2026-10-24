"""Train/evaluate Faster R-CNN or RetinaNet on the fixed MBDD2025 split.

Training selects checkpoints on validation only. The test split is accessible
only through --eval-only --split test, after a checkpoint has been selected.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    RetinaNet_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
    retinanet_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

from mbdd_data import (
    AUGMENTATION_CHOICES,
    CURATED_MANIFEST_SHA256,
    MBDDDataset,
    collate_fn,
)

try:  # fcntl is POSIX only; Windows uses the msvcrt fallback in acquire_run_lock
    import fcntl
except ImportError:
    fcntl = None


def acquire_run_lock(path: Path):
    """Stop two processes writing one run directory. Returns the held stream."""
    stream = path.open("a+")
    try:
        if fcntl is not None:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            import msvcrt
            stream.write("lock\n")
            stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as error:
        stream.close()
        raise RuntimeError(f"Another process is training in {path.parent}") from error
    return stream
from metrics import collect_predictions, ground_truth, score_predictions, write_json


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["fasterrcnn", "retinanet"], required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    # Training and evaluation use the curated split. The manifest next to the
    # curation scripts describes the original source and is deliberately not
    # the default here.
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).resolve().parents[1] /
                        "MBDD2025_curated_v1/split_manifest.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--augmentation", choices=AUGMENTATION_CHOICES, default="none",
                        help="Training-split augmentation; 'none' reproduces the baselines")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--checkpoint-every-minutes", type=float, default=15,
                        help="Save last.pt during long epochs at this interval")
    parser.add_argument("--checkpoint-every-batches", type=int, default=1000,
                        help="Also save last.pt after this many batches")
    parser.add_argument("--weights-path", type=Path,
                        help="Local COCO detector state_dict; avoids a download on compute nodes")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="For smoke tests only; main comparison requires COCO initialization")
    parser.add_argument("--resume", type=Path, help="Resume a last.pt checkpoint")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint for --eval-only")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--limit-train", type=int, help="Smoke test only")
    parser.add_argument("--limit-val", type=int, help="Smoke test only")
    args = parser.parse_args()
    if args.image_size < 256 or args.image_size % 32:
        parser.error("--image-size must be at least 256 and divisible by 32")
    if args.eval_only and not args.checkpoint:
        parser.error("--eval-only requires --checkpoint")
    if args.eval_only and args.augmentation != "none":
        parser.error("Augmentation applies to training only; evaluation is never augmented")
    if not args.eval_only and args.split != "val":
        parser.error("Training only evaluates validation; use --eval-only for test")
    if args.no_pretrained and args.weights_path:
        parser.error("--no-pretrained and --weights-path conflict")
    if args.eval_every < 1 or args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs, eval-every, batch-size and workers are invalid")
    if args.checkpoint_every_minutes <= 0 or args.checkpoint_every_batches <= 0:
        parser.error("Checkpoint intervals must be positive")
    return args


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_model(name: str, image_size: int, weights_path: Path | None,
                no_pretrained: bool):
    # Explicitly unfreeze the same backbone layers for fresh, resumed and
    # evaluation construction, regardless of whether weights are loaded now.
    kwargs = {"min_size": image_size, "max_size": image_size,
              "trainable_backbone_layers": 5}
    if name == "fasterrcnn":
        kwargs.update(box_score_thresh=0.001, box_nms_thresh=0.50,
                      box_detections_per_img=100)
        weights = None if no_pretrained or weights_path else FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(weights=weights, weights_backbone=None,
                                          num_classes=91 if weights_path else None, **kwargs)
        if weights_path:
            model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 6)
    else:
        kwargs.update(score_thresh=0.001, nms_thresh=0.50, detections_per_img=100)
        weights = None if no_pretrained or weights_path else RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        model = retinanet_resnet50_fpn_v2(weights=weights, weights_backbone=None,
                                         num_classes=91 if weights_path else None, **kwargs)
        if weights_path:
            model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        anchors = model.anchor_generator.num_anchors_per_location()[0]
        model.head.classification_head = RetinaNetClassificationHead(
            model.backbone.out_channels, anchors, 6,
            norm_layer=partial(nn.GroupNorm, 32))
    return model


def save_checkpoint(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


STOP_REQUESTED = False


def request_stop(signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"Received signal {signum}; saving a resumable checkpoint after this batch", flush=True)


def training_payload(model, optimizer, scheduler, scaler, args, manifest_sha256,
                     epoch, batches_done, loss_sum, loss_count, global_step, best,
                     pending_validation_epoch=None):
    return {
        "checkpoint_format": 2,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
        "epoch": epoch, "batch_in_epoch": batches_done,
        "loss_sum": loss_sum, "loss_count": loss_count,
        "global_step": global_step, "best_map": best,
        "pending_validation_epoch": pending_validation_epoch,
        "model_name": args.model, "image_size": args.image_size,
        "batch_size": args.batch_size, "seed": args.seed, "learning_rate": args.lr,
        "augmentation": args.augmentation,
        "manifest_sha256": manifest_sha256, "rng": rng_state(),
    }


def run_validation(model, loader, dataset, device, output_dir: Path, tag: str):
    predictions, timing = collect_predictions(model, loader, device, dataset)
    metrics = score_predictions(ground_truth(dataset), predictions)
    metrics["timing"] = timing
    metrics["split"] = dataset.split
    metrics["manifest_sha256"] = dataset.manifest_sha256
    write_json(output_dir / f"{tag}_predictions.json", predictions)
    write_json(output_dir / f"{tag}_metrics.json", metrics)
    return metrics


def evaluate_epoch(model, loader, dataset, device, output_dir: Path,
                   epoch: int, best: float, payload: dict) -> float:
    metrics = run_validation(model, loader, dataset, device, output_dir,
                             f"val_epoch_{epoch}")
    score = metrics["mAP50_95"]
    if score > best:
        best = score
        payload["best_map"] = best
        payload["pending_validation_epoch"] = None
        save_checkpoint(output_dir / "best.pt", payload)
        weights_file = output_dir / "best_weights.pt"
        save_checkpoint(weights_file, model.state_dict())
        metrics["model_file_mb"] = weights_file.stat().st_size / (1024 ** 2)
        write_json(output_dir / f"val_epoch_{epoch}_metrics.json", metrics)
    print(f"epoch={epoch} val_mAP50_95={score:.5f} best={best:.5f}", flush=True)
    return best


def main():
    args = arguments()
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required; submit this script through the CETUS GPU queue")
    device = torch.device("cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    selected_split = args.split if args.eval_only else "val"
    dataset = MBDDDataset(args.data_root, args.manifest, selected_split,
                          args.image_size, args.limit_val)
    curated_manifest = Path(__file__).resolve().parents[1] / "MBDD2025_curated_v1/split_manifest.csv"
    if args.manifest.resolve() == curated_manifest.resolve() and dataset.manifest_sha256 != CURATED_MANIFEST_SHA256:
        raise ValueError("Curated split manifest bytes have changed")
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=args.workers, collate_fn=collate_fn,
                        pin_memory=True)
    # A full checkpoint already contains every parameter; do not fetch COCO
    # weights again when evaluating or resuming it.
    restoring = args.eval_only or args.resume is not None
    model = build_model(args.model, args.image_size,
                        None if restoring else args.weights_path,
                        args.no_pretrained or restoring)
    model.to(device)
    if args.eval_only:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if checkpoint["model_name"] != args.model or checkpoint["image_size"] != args.image_size:
            raise ValueError("Checkpoint model or image size does not match command")
        if checkpoint["manifest_sha256"] != dataset.manifest_sha256:
            raise ValueError("Checkpoint was trained with a different split manifest")
        model.load_state_dict(checkpoint["model"])
        result = run_validation(model, loader, dataset, device, args.output, selected_split)
        print(json.dumps({"split": selected_split, "mAP50": result["mAP50"],
                          "mAP50_95": result["mAP50_95"]}, indent=2))
        return

    # The operating system releases this lock even if the process is killed,
    # preventing two jobs from writing the same last.pt concurrently.
    lock_stream = acquire_run_lock(args.output / ".train.lock")
    for name in ("SIGTERM", "SIGINT", "SIGUSR1"):
        handled = getattr(signal, name, None)  # SIGUSR1 does not exist on Windows
        if handled is not None:
            signal.signal(handled, request_stop)

    train_set = MBDDDataset(args.data_root, args.manifest, "train",
                            args.image_size, args.limit_train,
                            augment=args.augmentation)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    scaler = torch.amp.GradScaler("cuda")
    start_epoch = 1
    resume_batch = 0
    resume_loss_sum = 0.0
    resume_loss_count = 0
    global_step = 0
    best = -1.0
    pending_validation_epoch = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint["model_name"] != args.model or checkpoint["image_size"] != args.image_size:
            raise ValueError("Resume checkpoint does not match this model configuration")
        if checkpoint["manifest_sha256"] != train_set.manifest_sha256:
            raise ValueError("Resume checkpoint uses another manifest")
        for key, expected in (("batch_size", args.batch_size), ("seed", args.seed),
                              ("learning_rate", args.lr),
                              ("augmentation", args.augmentation)):
            if key in checkpoint and checkpoint[key] != expected:
                raise ValueError(f"Resume checkpoint {key} differs from this run")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        if checkpoint.get("checkpoint_format") == 2:
            start_epoch = checkpoint["epoch"]
            resume_batch = checkpoint["batch_in_epoch"]
            resume_loss_sum = checkpoint["loss_sum"]
            resume_loss_count = checkpoint["loss_count"]
            global_step = checkpoint["global_step"]
            pending_validation_epoch = checkpoint.get("pending_validation_epoch")
            restore_rng(checkpoint["rng"])
        else:
            start_epoch = checkpoint["epoch"] + 1
        best = checkpoint["best_map"]

    run_info = {
        "model": args.model, "seed": args.seed, "epochs": args.epochs,
        "batch_size": args.batch_size, "image_size": args.image_size,
        "learning_rate": args.lr, "augmentation": args.augmentation,
        "pretraining": "none" if args.no_pretrained else "COCO",
        "weights_path": str(args.weights_path) if args.weights_path else None,
        "manifest_sha256": train_set.manifest_sha256,
        "train_images": len(train_set), "validation_images": len(dataset),
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "pbs_requested_mem": os.environ.get("PBS_RESC_MEM"),
        "checkpoint_every_minutes": args.checkpoint_every_minutes,
        "checkpoint_every_batches": args.checkpoint_every_batches,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    write_json(args.output / "run.json", run_info)
    if pending_validation_epoch is not None:
        print(f"Completing pending validation for epoch {pending_validation_epoch}", flush=True)
        payload = training_payload(model, optimizer, scheduler, scaler, args,
                                   train_set.manifest_sha256, start_epoch, 0,
                                   0.0, 0, global_step, best,
                                   pending_validation_epoch)
        best = evaluate_epoch(model, loader, dataset, device, args.output,
                              pending_validation_epoch, best, payload)
        payload["best_map"] = best
        payload["pending_validation_epoch"] = None
        payload["rng"] = rng_state()
        save_checkpoint(args.output / "last.pt", payload)
        if STOP_REQUESTED:
            return 99
    if start_epoch > args.epochs:
        print(f"Training is already complete through epoch {args.epochs}; "
              f"selected checkpoint: {args.output / 'best.pt'}", flush=True)
        return 0
    last_checkpoint_time = time.monotonic()
    for epoch in range(start_epoch, args.epochs + 1):
        generator = torch.Generator().manual_seed(args.seed + epoch)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                  generator=generator, num_workers=args.workers,
                                  collate_fn=collate_fn, pin_memory=True,
                                  persistent_workers=args.workers > 0)
        if epoch == start_epoch and resume_batch > len(train_loader):
            raise ValueError("Resume batch count exceeds current epoch length")
        model.train()
        epoch_start = time.perf_counter()
        loss_sum = resume_loss_sum if epoch == start_epoch else 0.0
        loss_count = resume_loss_count if epoch == start_epoch else 0
        batches_done = resume_batch if epoch == start_epoch else 0
        if batches_done:
            print(f"Resuming epoch {epoch} after {batches_done} batches", flush=True)
        for batch_index, (images, targets) in enumerate(train_loader, 1):
            if batch_index <= batches_done:
                continue
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in target.items()}
                       for target in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss_terms = model(images, targets)
                loss = sum(loss_terms.values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}: {loss_terms}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach())
            loss_count += 1
            global_step += 1
            batches_done = batch_index
            elapsed = (time.monotonic() - last_checkpoint_time) / 60
            if (STOP_REQUESTED or elapsed >= args.checkpoint_every_minutes or
                    batch_index % args.checkpoint_every_batches == 0):
                payload = training_payload(model, optimizer, scheduler, scaler, args,
                                           train_set.manifest_sha256, epoch, batches_done,
                                           loss_sum, loss_count, global_step, best)
                save_checkpoint(args.output / "last.pt", payload)
                last_checkpoint_time = time.monotonic()
                print(f"checkpoint epoch={epoch} batch={batches_done}/{len(train_loader)} "
                      f"step={global_step}", flush=True)
            if STOP_REQUESTED:
                print("Stopped after saving last.pt; resubmit the same run name to resume", flush=True)
                return 99
        if STOP_REQUESTED:
            payload = training_payload(model, optimizer, scheduler, scaler, args,
                                       train_set.manifest_sha256, epoch, batches_done,
                                       loss_sum, loss_count, global_step, best)
            save_checkpoint(args.output / "last.pt", payload)
            return 99
        scheduler.step()
        print(f"epoch={epoch} mean_loss={loss_sum/loss_count:.5f} "
              f"seconds={time.perf_counter()-epoch_start:.1f}", flush=True)
        validation_due = epoch % args.eval_every == 0 or epoch == args.epochs
        payload = training_payload(model, optimizer, scheduler, scaler, args,
                                   train_set.manifest_sha256, epoch + 1, 0,
                                   0.0, 0, global_step, best,
                                   epoch if validation_due else None)
        # Persist the completed epoch before validation, which can be long.
        # A resubmission finishes any pending validation before the next epoch.
        save_checkpoint(args.output / "last.pt", payload)
        if STOP_REQUESTED:
            return 99
        if validation_due:
            best = evaluate_epoch(model, loader, dataset, device, args.output,
                                  epoch, best, payload)
            payload["best_map"] = best
            payload["pending_validation_epoch"] = None
            payload["rng"] = rng_state()
            save_checkpoint(args.output / "last.pt", payload)
            if STOP_REQUESTED:
                return 99
        last_checkpoint_time = time.monotonic()
    print(f"Selected checkpoint: {args.output / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
