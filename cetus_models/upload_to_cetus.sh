#!/usr/bin/env bash
# Run on the local Mac from the project directory or any directory.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CETUS_REMOTE:?Set CETUS_REMOTE to your CETUS user@host before uploading}"
REMOTE="$CETUS_REMOTE"
DEST="${CETUS_PROJECT:-mbdd2025_project}"
[[ "$DEST" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'CETUS_PROJECT must be a simple directory name' >&2; exit 2; }
[[ -f MBDD2025_curated_v1/split_manifest.csv ]] || { echo 'Curated dataset missing' >&2; exit 2; }
[[ -d MBDD2025_curated_v1/JPEGImages && -d MBDD2025_curated_v1/Annotations ]] || {
  echo 'Curated images or annotations missing' >&2; exit 2;
}
LOCAL_HASH="$(shasum -a 256 MBDD2025_curated_v1/split_manifest.csv | awk '{print $1}')"
[[ "$LOCAL_HASH" == 6f4ab573cd61f1a28120120b10126feda16635e98a0d7db22270dcaba82007c1 ]] || {
  echo "Unexpected split manifest hash: $LOCAL_HASH" >&2; exit 2;
}
ssh "$REMOTE" "mkdir -p ~/$DEST/cetus_models ~/$DEST/MBDD2025_curated_v1"
rsync -avP --exclude '__pycache__/' --exclude '*.pyc' \
  cetus_models/ "$REMOTE:~/$DEST/cetus_models/"
rsync -avP MBDD2025_curated_v1/ "$REMOTE:~/$DEST/MBDD2025_curated_v1/"
ssh "$REMOTE" "cd ~/$DEST && sha256sum MBDD2025_curated_v1/split_manifest.csv"
printf 'Upload complete. Verify the remote hash equals %s\n' "$LOCAL_HASH"
