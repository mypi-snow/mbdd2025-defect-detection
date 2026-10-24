# Curated dataset placement

The tracked `split_manifest.csv` is the exact 12,026-image, scene-grouped split
used for both published detector runs. Its SHA-256 is
`6f4ab573cd61f1a28120120b10126feda16635e98a0d7db22270dcaba82007c1`.

The JPEGs, VOC XML annotations, and YOLO text labels are intentionally absent
from Git. Obtain the original [MBDD2025 dataset](https://doi.org/10.5281/zenodo.15622584)
and reproduce the curated derivative with `cetus_models/materialize_curated_dataset.py`
using the inputs under `curation/audit_inputs/`. The team's separate curated
archive can instead be extracted here; verify that its manifest has the hash
above. The expected directories are `JPEGImages/`, `Annotations/`, and `Labels/`.

Do not re-randomise the split when comparing models. Keep `test` sealed until
the model and settings have been chosen on validation.
