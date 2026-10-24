#!/usr/bin/env bash
# Run on a CETUS login node, before submitting any GPU job.
set -euo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT="$PWD"
python_ok() {
  "$1" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)' \
    >/dev/null 2>&1
}

# CETUS's default python3 may be too old. Prefer a supported installed
# interpreter; otherwise install a private Python 3.11 without changing shell
# startup files or touching system Python.
if [[ -n "${PYTHON:-}" ]]; then
  python_ok "$PYTHON" || { echo "PYTHON=$PYTHON must be Python 3.10–3.13" >&2; exit 2; }
elif [[ -x .venv/bin/python ]] && python_ok .venv/bin/python; then
  PYTHON="$PROJECT_ROOT/.venv/bin/python"
else
  PYTHON=""
  for candidate in python3.11 python3.12 python3.10 python3.13 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && python_ok "$candidate"; then
      PYTHON="$(command -v "$candidate")"
      break
    fi
  done
fi

if [[ -z "$PYTHON" ]]; then
  [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || {
    echo 'Automatic Python bootstrap requires Linux x86_64; set PYTHON to a compatible interpreter' >&2
    exit 2
  }
  if [[ -e .venv ]]; then
    echo 'An incompatible .venv exists. Rename it before retrying setup.' >&2
    exit 2
  fi
  if [[ ! -x .miniforge/bin/mamba ]]; then
    [[ ! -e .miniforge ]] || { echo 'Incomplete .miniforge exists; inspect or rename it' >&2; exit 2; }
    version='26.7.2-0'
    asset="Miniforge3-${version}-Linux-x86_64.sh"
    base_url="https://github.com/conda-forge/miniforge/releases/download/${version}"
    mkdir -p downloads
    echo 'No suitable Python found; downloading a private Miniforge installer'
    curl -fL --retry 3 -o "downloads/$asset" "$base_url/$asset"
    curl -fL --retry 3 -o "downloads/$asset.sha256" "$base_url/$asset.sha256"
    (cd downloads && sha256sum -c "$asset.sha256")
    bash "downloads/$asset" -b -p "$PROJECT_ROOT/.miniforge"
  fi
  "$PROJECT_ROOT/.miniforge/bin/mamba" create -y -p "$PROJECT_ROOT/.venv" python=3.11 pip
  PYTHON="$PROJECT_ROOT/.venv/bin/python"
fi
echo "Using Python: $PYTHON ($("$PYTHON" --version 2>&1))"
"$PYTHON" - <<'PY'
import csv
import hashlib
import sys
from pathlib import Path
if not (3, 10) <= sys.version_info[:2] <= (3, 13):
    raise SystemExit('Python 3.10–3.13 is required')
root = Path('MBDD2025_curated_v1')
manifest = root / 'split_manifest.csv'
expected = '6f4ab573cd61f1a28120120b10126feda16635e98a0d7db22270dcaba82007c1'
if not manifest.is_file() or hashlib.sha256(manifest.read_bytes()).hexdigest() != expected:
    raise SystemExit('Missing or changed curated split manifest; upload dataset again')
with manifest.open(newline='', encoding='utf-8-sig') as stream:
    rows = list(csv.DictReader(stream))
missing = [(sub, row['ImageID']) for row in rows for sub, ext in (('JPEGImages', '.jpg'), ('Annotations', '.xml'))
           if not (root / sub / (row['ImageID'] + ext)).is_file()]
if missing:
    raise SystemExit(f'Incomplete dataset upload: first missing files: {missing[:5]}')
print(f'Dataset files verified: {len(rows)} image/XML pairs')
PY
if [[ ! -x .venv/bin/python ]]; then
  "$PYTHON" -m venv .venv
fi
PIP="$PROJECT_ROOT/.venv/bin/python"
python_ok "$PIP" || { echo 'Existing .venv has an unsupported Python; rename it and rerun setup' >&2; exit 2; }
mkdir -p wheelhouse weights
# A recent pip recognises the Linux CUDA wheels. Download it first and then
# install it from the wheelhouse, before resolving the training packages.
"$PIP" -m pip download --dest wheelhouse 'pip==25.2'
"$PIP" -m pip install --no-index --find-links wheelhouse 'pip==25.2'
# Download every dependency before installing anything. The pinned CUDA 11.8
# wheels match PyTorch's published torch/vision version pairing.
"$PIP" -m pip download --dest wheelhouse --index-url https://download.pytorch.org/whl/cu118 'torch==2.7.1' 'torchvision==0.22.1'
"$PIP" -m pip download --dest wheelhouse -r cetus_models/requirements.txt
"$PIP" -m pip install --no-index --find-links wheelhouse 'torch==2.7.1' 'torchvision==0.22.1' -r cetus_models/requirements.txt
if ! "$PIP" -c 'import torch, torchvision, pycocotools, PIL, numpy' >/dev/null 2>&1; then
  echo 'Python packages appear incomplete; reinstalling from the downloaded wheelhouse'
  "$PIP" -m pip install --no-index --find-links wheelhouse --force-reinstall \
    'torch==2.7.1' 'torchvision==0.22.1' -r cetus_models/requirements.txt
fi
"$PIP" - <<'PY'
import torch, torchvision, pycocotools, PIL, numpy
assert torch.version.cuda is not None, 'CPU-only PyTorch installed'
print('Installed:', torch.__version__, torchvision.__version__, 'CUDA build:', torch.version.cuda)
PY
"$PIP" -m pip check
"$PIP" -m pip freeze > environment.lock.txt
for model in fasterrcnn retinanet; do
  if [[ ! -s "weights/${model}_coco_v2.pt" ]]; then
    "$PIP" cetus_models/prefetch_weights.py --model "$model" --output-dir weights
  fi
done
"$PIP" - <<'PY'
from pathlib import Path
import torch
for model in ('fasterrcnn', 'retinanet'):
    path = Path('weights') / f'{model}_coco_v2.pt'
    state = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not state:
        raise SystemExit(f'Invalid weight file: {path}; remove it and rerun setup')
    print('Verified weight file:', path)
PY
sha256sum weights/*_coco_v2.pt > weights/SHA256SUMS
printf 'Ready: %s\n' "$PROJECT_ROOT"
