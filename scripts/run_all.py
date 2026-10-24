#!/usr/bin/env python3
"""Run the full MBDD2025 augmentation experiment, unattended and crash safe.

Chain, in order:
    1. Faster R-CNN  baseline   (augmentation: none)
    2. Faster R-CNN  augmented  (augmentation: standard)
    3. RetinaNet     baseline   (augmentation: none)
    4. RetinaNet     augmented  (augmentation: standard)
    5. Held-out test evaluation for each of the four

Safe to stop and re-run at any time. Completed jobs are skipped, a job that
died is resumed from its last checkpoint, and a job already running in another
window is waited for rather than started twice.

    python scripts/run_all.py              run the chain
    python scripts/run_all.py --status     print what is done and what is left
    python scripts/run_all.py --dry-run    print the commands without running them
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "cetus_models"
DATA = ROOT / "MBDD2025_curated_v1"   # place the curated images and annotations here
RUNS = ROOT / "runs"

EPOCHS = 40
COMMON = ["--epochs", str(EPOCHS), "--workers", "6", "--seed", "42",
          "--batch-size", "1", "--image-size", "640", "--eval-every", "5"]

# (run name, model, augmentation). Order is the execution order.
JOBS = [
    ("frcnn_local_base", "fasterrcnn", "none"),
    ("frcnn_local_aug", "fasterrcnn", "standard"),
    ("retinanet_local_base", "retinanet", "none"),
    ("retinanet_local_aug", "retinanet", "standard"),
]

MAX_ATTEMPTS = 6
MIN_HEALTHY_SECONDS = 180   # a restart that dies faster than this counts as a crash loop
POLL_SECONDS = 60


def log(message: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    try:   # a copy on disk, so progress is visible without this terminal
        RUNS.mkdir(parents=True, exist_ok=True)
        with (RUNS / "run_all.log").open("a", encoding="utf8") as stream:
            stream.write(line + "\n")
    except OSError:
        pass


def train_log(name: str) -> Path:
    return RUNS / name / "train.log"


def is_complete(name: str) -> bool:
    path = train_log(name)
    if not path.exists():
        return False
    return "Selected checkpoint:" in path.read_text(encoding="utf8", errors="replace")


def test_done(name: str) -> bool:
    return (RUNS / name / "test_eval" / "test_metrics.json").is_file()


def is_running(name: str) -> bool:
    """True if some python process elsewhere is already working on this run."""
    if sys.platform != "win32":
        out = subprocess.run(["bash", "-lc", f"ps -ef | grep -c '[t]rain_detector.*{name}'"],
                             capture_output=True, text=True)
        return out.stdout.strip() not in ("", "0")
    query = (
        "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        f"Where-Object {{ $_.CommandLine -like '*{name}*' }}).Count"
    )
    out = subprocess.run(["powershell", "-NoProfile", "-Command", query],
                         capture_output=True, text=True, creationflags=NO_WINDOW)
    try:
        return int(out.stdout.strip()) > 0
    except ValueError:
        return False


def train_command(name: str, model: str, augmentation: str) -> list[str]:
    command = [sys.executable, "-u", "train_detector.py",
               "--model", model,
               "--data-root", str(DATA),
               "--manifest", str(DATA / "split_manifest.csv"),
               "--output", str(RUNS / name),
               "--augmentation", augmentation, *COMMON]
    if (RUNS / name / "last.pt").is_file():
        command += ["--resume", str(RUNS / name / "last.pt")]
    return command


def test_command(name: str, model: str) -> list[str]:
    return [sys.executable, "-u", "train_detector.py",
            "--model", model,
            "--data-root", str(DATA),
            "--manifest", str(DATA / "split_manifest.csv"),
            "--output", str(RUNS / name / "test_eval"),
            "--eval-only", "--split", "test",
            "--checkpoint", str(RUNS / name / "best.pt"),
            "--image-size", "640", "--workers", "6"]


# Children get their own console-free process group, so closing a terminal
# cannot deliver CTRL_CLOSE_EVENT to a training run and abort it.
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0


def execute(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf8") as stream:
        stream.write(f"\n=== {datetime.now():%Y-%m-%d %H:%M:%S} "
                     f"{' '.join(command)} ===\n")
        stream.flush()
        return subprocess.run(command, cwd=SCRIPTS, stdout=stream,
                              stderr=subprocess.STDOUT,
                              creationflags=NO_WINDOW).returncode


def wait_for_foreign(name: str) -> None:
    log(f"{name}: already running in another window, waiting for it")
    while is_running(name):
        time.sleep(POLL_SECONDS)
    log(f"{name}: that process has exited")


def run_training(name: str, model: str, augmentation: str, dry: bool) -> bool:
    if is_complete(name):
        log(f"{name}: already complete, skipping")
        return True
    if dry:
        print("  " + " ".join(train_command(name, model, augmentation)))
        return True
    if is_running(name):
        wait_for_foreign(name)
        if is_complete(name):
            log(f"{name}: complete")
            return True

    short_runs = 0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        resuming = (RUNS / name / "last.pt").is_file()
        log(f"{name}: {'resuming from last.pt' if resuming else 'starting fresh'} "
            f"(attempt {attempt} of {MAX_ATTEMPTS})")
        started = time.monotonic()
        code = execute(train_command(name, model, augmentation), train_log(name))
        elapsed = time.monotonic() - started
        log(f"{name}: exited with code {code} after {timedelta(seconds=int(elapsed))}")
        if is_complete(name):
            log(f"{name}: COMPLETE")
            return True
        if elapsed < MIN_HEALTHY_SECONDS:
            short_runs += 1
            if short_runs >= 3:
                log(f"{name}: CRASH LOOP, three restarts died within "
                    f"{MIN_HEALTHY_SECONDS}s each. Stopping.")
                return False
        else:
            short_runs = 0
        time.sleep(30)   # let the GPU and the run lock clear
    log(f"{name}: gave up after {MAX_ATTEMPTS} attempts")
    return False


def run_test(name: str, model: str, dry: bool) -> bool:
    if test_done(name):
        log(f"{name}: test evaluation already done, skipping")
        return True
    if not (RUNS / name / "best.pt").is_file():
        log(f"{name}: no best.pt, cannot evaluate on test")
        return False
    if dry:
        print("  " + " ".join(test_command(name, model)))
        return True
    log(f"{name}: evaluating on the held-out test split")
    code = execute(test_command(name, model), RUNS / name / "test_eval" / "test.log")
    ok = test_done(name)
    log(f"{name}: test evaluation {'done' if ok else f'FAILED (code {code})'}")
    return ok


VALIDATION_OVERHEAD = 45.0    # COCO scoring, JSON write and checkpoint save
RETINANET_FACTOR = 1.08       # RetinaNet trains ~8% slower than Faster R-CNN
DEFAULT_EPOCH_SECONDS = 1150.0


def hm(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m"


def read_progress(name: str) -> dict:
    """Parse a run log into epochs done, position in the current epoch, and pace."""
    info = {"epochs_done": 0, "batch": 0, "batches": 0, "epoch_seconds": [],
            "val_seconds": None, "last_activity": None}
    path = train_log(name)
    if not path.exists():
        return info
    info["last_activity"] = time.time() - path.stat().st_mtime
    text = path.read_text(encoding="utf8", errors="replace")
    done = [(int(e), float(s)) for e, s in
            re.findall(r"^epoch=(\d+) mean_loss=[\d.]+ seconds=([\d.]+)", text, re.M)]
    if done:
        info["epochs_done"] = max(e for e, _ in done)
        info["epoch_seconds"] = [s for _, s in done]
    checkpoints = re.findall(r"^checkpoint epoch=(\d+) batch=(\d+)/(\d+)", text, re.M)
    if checkpoints:
        epoch, batch, total = (int(v) for v in checkpoints[-1])
        info["batches"] = total
        info["batch"] = batch if epoch > info["epochs_done"] else 0
    metrics = sorted((RUNS / name).glob("val_epoch_*_metrics.json")) if (RUNS / name).is_dir() else []
    if metrics:
        timing = json.loads(metrics[-1].read_text()).get("timing", {})
        fps = timing.get("sequential_fps")
        images = json.loads(metrics[-1].read_text()).get("image_count", 0)
        if fps and images:
            info["val_seconds"] = images / fps + VALIDATION_OVERHEAD
    return info


def pace(name: str, model: str, measured: list[float]) -> float:
    if measured:
        return sum(measured) / len(measured)
    for other, other_model, _aug in JOBS:          # borrow from a finished run
        if other_model == model and other != name:
            seen = read_progress(other)["epoch_seconds"]
            if seen:
                return sum(seen) / len(seen)
    reference = DEFAULT_EPOCH_SECONDS
    for other, other_model, _aug in JOBS:
        seen = read_progress(other)["epoch_seconds"]
        if seen:
            reference = sum(seen) / len(seen)
            if other_model == "retinanet":
                reference /= RETINANET_FACTOR
            break
    return reference * (RETINANET_FACTOR if model == "retinanet" else 1.0)


def remaining_seconds(name: str, model: str, info: dict) -> float:
    if is_complete(name):
        return 0.0
    epoch_s = pace(name, model, info["epoch_seconds"])
    fraction = info["batch"] / info["batches"] if info["batches"] else 0.0
    epochs_left = EPOCHS - info["epochs_done"] - fraction
    val_s = info["val_seconds"] or (1804 / 13.4 + VALIDATION_OVERHEAD)
    vals_left = sum(1 for e in range(info["epochs_done"] + 1, EPOCHS + 1)
                    if e % 5 == 0 or e == EPOCHS)
    return max(0.0, epochs_left) * epoch_s + vals_left * val_s


def status() -> None:
    print(f"\nMBDD2025 chain status        {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    header = f"{'run':<23}{'state':<11}{'progress':<26}{'best mAP50_95':>14}{'remaining':>12}"
    print(header)
    print("-" * len(header))
    total_left = 0.0
    active = None
    for name, model, _aug in JOBS:
        info = read_progress(name)
        complete = is_complete(name)
        fresh = info["last_activity"] is not None and info["last_activity"] < 300
        if complete:
            state = "complete"
        elif fresh:
            state = "RUNNING"
            active = (name, info)
        elif info["epochs_done"] or (RUNS / name / "last.pt").is_file():
            state = "stalled"
        else:
            state = "queued"
        if complete:
            progress = f"{EPOCHS}/{EPOCHS} epochs"
        elif info["epochs_done"] or info["batch"]:
            progress = f"epoch {info['epochs_done'] + 1}/{EPOCHS}"
            if info["batches"]:
                progress += f", batch {info['batch']}/{info['batches']}"
        else:
            progress = "-"
        best = "-"
        files = sorted((RUNS / name).glob("val_epoch_*_metrics.json")) if (RUNS / name).is_dir() else []
        if files:
            best = f"{max(json.loads(p.read_text())['mAP50_95'] for p in files):.4f}"
        left = remaining_seconds(name, model, info)
        total_left += left
        print(f"{name:<23}{state:<11}{progress:<26}{best:>14}{(hm(left) if left else '-'):>12}")

    pending_tests = [n for n, _m, _a in JOBS if not test_done(n)]
    test_left = len(pending_tests) * (1804 / 13.4 + VALIDATION_OVERHEAD)
    total_left += test_left
    print(f"{'test evaluations':<23}{'queued':<11}{f'{len(pending_tests)} pending':<26}{'-':>14}{hm(test_left):>12}")

    print()
    if active:
        name, info = active
        print(f"current job      {name}, finishes about "
              f"{datetime.now() + timedelta(seconds=remaining_seconds(name, dict((n, m) for n, m, _ in JOBS)[name], info)):%a %d %b %H:%M}")
        if info["epoch_seconds"]:
            print(f"measured pace    {sum(info['epoch_seconds']) / len(info['epoch_seconds']):.0f}s per epoch")
        print(f"last log write   {int(info['last_activity'])}s ago")
    else:
        print("current job      nothing is training right now")
    print(f"whole chain      about {hm(total_left)} left, finishing "
          f"{datetime.now() + timedelta(seconds=total_left):%a %d %b %H:%M}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--status", action="store_true", help="print progress and exit")
    parser.add_argument("--dry-run", action="store_true", help="print commands only")
    args = parser.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)
    if not args.status:
        log(f"run_all.py invoked by {sys.executable}")
    for path, label in ((SCRIPTS / "train_detector.py", "training script"),
                        (DATA / "split_manifest.csv", "split manifest"),
                        (DATA / "JPEGImages", "images")):
        if not path.exists():
            message = f"Missing {label}: {path}"
            print(message, file=sys.stderr)
            log("ABORTED - " + message)
            return 2

    if args.status:
        status()
        return 0

    log(f"chain starting: {len(JOBS)} training runs then {len(JOBS)} test evaluations")
    for name, model, augmentation in JOBS:
        if not run_training(name, model, augmentation, args.dry_run):
            log(f"STOPPING: {name} did not complete. Fix it, then run this script again.")
            return 1
    for name, model, _aug in JOBS:
        run_test(name, model, args.dry_run)
    log("chain finished")
    status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
