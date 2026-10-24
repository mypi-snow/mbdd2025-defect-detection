# Running the MBDD2025 baselines on UTS CETUS

The checked-in scripts run Faster R-CNN ResNet-50 FPN V2 and RetinaNet
ResNet-50 FPN V2 on the **fixed curated split**. They do not implement the
teammate's MATLAB tiny-YOLOv4 model, DETR, or augmentation experiments.

## 1. Place the dataset and upload

Extract the curated dataset into `MBDD2025_curated_v1/` at the repository root.
The manifest SHA-256 must be
`6f4ab573cd61f1a28120120b10126feda16635e98a0d7db22270dcaba82007c1`.
From your own computer, set your CETUS login and run the resumable upload:

```bash
export CETUS_REMOTE='your-uts-id@hpc-login01.hpc.uts.edu.au'
bash cetus_models/upload_to_cetus.sh
ssh "$CETUS_REMOTE"
```

The upload uses `rsync` and verifies the remote manifest hash. It sends code
and the extracted curated dataset, not the Git repository metadata or saved
results. `CETUS_PROJECT` can override the default remote directory name
`mbdd2025_project`.

## 2. Prepare the environment on a CETUS login node

```bash
cd ~/mbdd2025_project
bash cetus_models/prepare_cetus.sh
```

Wait for the final `Ready:` line. The setup script selects Python 3.10–3.13,
downloads all wheels before installation, installs pinned CUDA 11.8 PyTorch
2.7.1 and torchvision 0.22.1, verifies imports, and prefetches official COCO
detector weights. It does not download or install packages in GPU jobs. If SSH
disconnects during preparation, rerun the same command and wait for `Ready:`.

## 3. Smoke test, train and resume

```bash
cd ~/mbdd2025_project
.venv/bin/python cetus_models/submit_cetus.py --model fasterrcnn --mode smoke --mem 32gb --email you@uts.edu.au
.venv/bin/python cetus_models/submit_cetus.py --model retinanet --mode smoke --mem 32gb --email you@uts.edu.au
qstat -u "$USER"
```

Smoke tests use 12 training and six validation images at 256 pixels for one
epoch; their metrics are pipeline checks only. Once both pass, use the exact
recorded baseline settings:

```bash
.venv/bin/python cetus_models/submit_cetus.py --model fasterrcnn --mode train --epochs 100 --image-size 640 --batch-size 1 --seed 42 --mem 64gb --email you@uts.edu.au
.venv/bin/python cetus_models/submit_cetus.py --model retinanet --mode train --epochs 100 --image-size 640 --batch-size 1 --seed 42 --mem 64gb --email you@uts.edu.au
```

Both jobs ask PBS for one GPU, six CPU cores, 64 GB CPU RAM and 48 hours.
GPU video memory is separate from PBS CPU RAM. Remove `--email` to use your
PBS account's default email. `--dry-run` prints the qsub request without
submitting it. The queue may be busy; `qstat -u "$USER"` shows queued and
running jobs.

The trainer writes `last.pt` throughout training and a validation-selected
`best.pt`. If a job hits walltime, resubmit the **same command** with the same
run name; the PBS script automatically resumes from `last.pt`, including
optimizer, scheduler and random-number state. Do not run two jobs with the
same run name simultaneously. For a changed image size, seed or batch size,
start a new `--run-name` instead.

Add `--augmentation standard` to a train command for the augmented arm. Its
default run name becomes `<model>_curated_standard_seed42`, so it cannot resume
the baseline's `last.pt`; pass that name as `--run-name` when testing it. The
default, `--augmentation none`, reproduces the baselines exactly. To run the
four-run comparison on a single local GPU instead, see `scripts/run_all.py`.

## 4. Evaluate after selection

```bash
.venv/bin/python cetus_models/submit_cetus.py --model fasterrcnn --mode test --mem 64gb --email you@uts.edu.au
.venv/bin/python cetus_models/submit_cetus.py --model retinanet --mode test --mem 64gb --email you@uts.edu.au
```

`test` reads `runs/<model>_curated_seed42/best.pt` and writes a separate
`test_eval/test_metrics.json`. Run this only after model/settings selection on
validation. The repository currently contains **validation**, not test,
results. The original runs and their learning curves are documented in the
repository [README](../README.md).

PBS joins stdout/stderr in the submission directory and sends begin, end and
abort mail. The job prints timestamps for start, handled termination and exit.
The checkpoint interval is 15 minutes or 1,000 batches by default. A sudden
node failure can still lose work since the last checkpoint.
