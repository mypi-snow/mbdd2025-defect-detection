#!/usr/bin/env python3
"""Validate the curated project and submit a PBS GPU job on UTS CETUS."""
from __future__ import annotations

import argparse
import hashlib
import math
import re
import subprocess
from pathlib import Path

EXPECTED_SHA256 = "6f4ab573cd61f1a28120120b10126feda16635e98a0d7db22270dcaba82007c1"
ROOT = Path(__file__).resolve().parent.parent
QUEUE = "small_gpuq"


def run_text(command: list[str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    return result.stdout


def bytes_from_pbs(value: str) -> int:
    match = re.fullmatch(r"\s*([\d.]+)\s*(b|kb|mb|gb|tb)\s*", value, re.I)
    if not match:
        raise ValueError(f"Unrecognised PBS memory value: {value!r}")
    factor = {"b": 1, "kb": 1024, "mb": 1024**2, "gb": 1024**3, "tb": 1024**4}[match.group(2).lower()]
    return int(float(match.group(1)) * factor)


def queue_max_memory(output: str) -> int | None:
    match = re.search(r"^\s*resources_max\.mem\s*=\s*(\S+)", output, re.M)
    value = bytes_from_pbs(match.group(1)) if match else None
    return value if value else None


def node_max_memory(output: str) -> int | None:
    # CETUS documents exec01..exec04 as the small GPU nodes. Require an
    # explicit GPU resource rather than considering CPU-only hosts.
    sizes = []
    for block in re.split(r"(?m)^(?=\S[^\n]*\n\s*(?:Mom|state)\s*=)", output):
        first = block.splitlines()[0].strip() if block.strip() else ""
        if not re.search(r"\bexec0[1-4]\b", first):
            continue
        gpu = re.search(r"^\s*resources_available\.ngpus\s*=\s*(\d+)", block, re.M)
        memory = re.search(r"^\s*resources_available\.mem\s*=\s*(\S+)", block, re.M)
        if gpu and int(gpu.group(1)) >= 1 and memory:
            size = bytes_from_pbs(memory.group(1))
            if size:
                sizes.append(size)
    return max(sizes) if sizes else None


def choose_memory(value: str) -> tuple[str, str]:
    if value != "auto":
        size = bytes_from_pbs(value)
        if size < 8 * 1024**3:
            raise ValueError("Request at least 8gb CPU RAM")
        return value.lower(), "manual override"
    queue_cap = None
    node_cap = None
    errors = []
    try:
        queue_cap = queue_max_memory(run_text(["qstat", "-Qf", QUEUE]))
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        errors.append(f"qstat: {error}")
    try:
        node_cap = node_max_memory(run_text(["pbsnodes", "-av"]))
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        errors.append(f"pbsnodes: {error}")
    caps = [cap for cap in (queue_cap, node_cap) if cap is not None]
    if not caps:
        raise RuntimeError("Could not discover a maximum from CETUS. "
                           "Run 'qstat -Qf small_gpuq' and 'pbsnodes -av', then "
                           "rerun with --mem 64gb (or another confirmed value). "
                           + "; ".join(errors))
    cap = min(caps)
    # PBS reports CPU RAM; leave OS/file-cache headroom on the node.
    request_gb = math.floor(cap / 1024**3) - 4
    if request_gb < 8:
        raise RuntimeError(f"Detected only {cap/1024**3:.1f} GiB allocatable memory")
    basis = f"queue={queue_cap/1024**3:.1f} GiB" if queue_cap else "queue=unlimited/unspecified"
    basis += f", GPU node={node_cap/1024**3:.1f} GiB" if node_cap else ", GPU node=unknown"
    return f"{request_gb}gb", basis + "; 4 GiB headroom"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["fasterrcnn", "retinanet"], required=True)
    parser.add_argument("--mode", choices=["smoke", "train", "test"], required=True)
    parser.add_argument("--run-name", help="Use the same name when resuming or testing")
    parser.add_argument("--mem", default="auto", help="CPU RAM request; auto probes CETUS, e.g. 64gb")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--augmentation", choices=("none", "standard"), default="none",
                        help="Training augmentation; changes the default run name")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--checkpoint-minutes", type=int, default=15)
    parser.add_argument("--checkpoint-batches", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--email", help="Optional PBS begin/end/abort notification address")
    args = parser.parse_args()
    if args.seed < 0 or args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.eval_every < 1:
        parser.error("seed, epochs, batch size, workers or eval interval invalid")
    if args.image_size < 256 or args.image_size % 32 or args.checkpoint_minutes < 1 or args.checkpoint_batches < 1:
        parser.error("image size must be >=256 and divisible by 32; checkpoint intervals must be positive")
    # An augmented run must not default onto the baseline directory, or the PBS
    # script would resume the baseline's last.pt instead of starting a new run.
    suffix = "" if args.augmentation == "none" else f"_{args.augmentation}"
    name = args.run_name or f"{args.model}_curated{suffix}_seed{args.seed}"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        parser.error("run name may contain only letters, digits, _ and -")
    manifest = ROOT / "MBDD2025_curated_v1/split_manifest.csv"
    if not manifest.is_file() or hashlib.sha256(manifest.read_bytes()).hexdigest() != EXPECTED_SHA256:
        parser.error("curated split manifest missing or changed; upload the correct dataset first")
    for subdir in ("JPEGImages", "Annotations"):
        if not (manifest.parent / subdir).is_dir():
            parser.error(f"curated {subdir} directory is missing")
    venv_python = ROOT / ".venv/bin/python"
    if not venv_python.is_file():
        parser.error("Python environment missing; run bash cetus_models/prepare_cetus.sh first")
    import_check = subprocess.run(
        [str(venv_python), "-c",
         "import torch, torchvision, pycocotools; "
         "assert torch.version.cuda is not None, 'CPU-only PyTorch installed'"],
        text=True, capture_output=True)
    if import_check.returncode:
        parser.error("Python environment is incomplete or broken. "
                     "Run bash cetus_models/prepare_cetus.sh and wait for 'Ready'. "
                     f"Import error: {import_check.stderr.strip()[-400:]}")
    out = ROOT / "runs" / name
    if args.mode == "train" and not (out / "last.pt").is_file():
        if not (ROOT / f"weights/{args.model}_coco_v2.pt").is_file():
            parser.error("COCO weights missing; run prepare_cetus.sh first")
    if args.mode == "test" and not (out / "best.pt").is_file():
        parser.error(f"best.pt missing in {out}; choose a validation-selected model first")
    if args.mode == "smoke" and args.mem == "auto":
        memory, basis = "32gb", "smoke-test default"
    else:
        memory, basis = choose_memory(args.mem)
    image_size = 256 if args.mode == "smoke" else args.image_size
    epochs = 1 if args.mode == "smoke" else args.epochs
    batch_size = 1 if args.mode == "smoke" else args.batch_size
    workers = 0 if args.mode == "smoke" else args.workers
    eval_every = 1 if args.mode == "smoke" else args.eval_every
    variables = {
        "MODEL": args.model, "MODE": args.mode, "RUN_NAME": name,
        "SEED": args.seed, "EPOCHS": epochs, "IMAGE_SIZE": image_size,
        "BATCH_SIZE": batch_size, "WORKERS": workers,
        "EVAL_EVERY": eval_every, "CHECKPOINT_MINUTES": args.checkpoint_minutes,
        "CHECKPOINT_BATCHES": args.checkpoint_batches,
        "AUGMENTATION": "none" if args.mode != "train" else args.augmentation,
    }
    walltime = "01:00:00" if args.mode == "smoke" else "48:00:00"
    command = ["qsub", "-l", f"select=1:ncpus=6:mem={memory}:ngpus=1",
               "-l", f"walltime={walltime}", "-v",
               ",".join(f"{key}={value}" for key, value in variables.items()),
               "cetus_models/cetus_gpu.pbs"]
    if args.email:
        if not re.fullmatch(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+", args.email):
            parser.error("--email must be a plain email address")
        command[1:1] = ["-M", args.email]
    print(f"CPU memory: {memory} ({basis})")
    print(f"Batch size: {batch_size}; image size: {image_size}. GPU VRAM is separate from PBS CPU RAM.")
    print(f"Run name: {name}; augmentation: {variables['AUGMENTATION']}")
    print("PBS command:", " ".join(command))
    if not args.dry_run:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
        print("Submitted job:", result.stdout.strip())
        print("PBS mail will report begin, end, and abort", f"to {args.email}" if args.email else "to the PBS account address")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
