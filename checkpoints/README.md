# Trained model checkpoints

The four full PyTorch checkpoints are distributed as assets on the GitHub
release associated with these results. They are kept out of Git history because
each file exceeds GitHub's 100 MB normal Git object limit. Download the assets
named `fasterrcnn_best.pt`, `fasterrcnn_last.pt`, `retinanet_best.pt`, and
`retinanet_last.pt`. Verify them with `shasum -a 256` against `SHA256SUMS`.

`best.pt` is the validation-selected epoch-25 checkpoint and includes the
model, optimizer, scheduler, loss scaler and random-number states. `last.pt`
holds the completed epoch-100 state for resumption or inspection. For CETUS
evaluation, place `fasterrcnn_best.pt` at
`runs/fasterrcnn_curated_seed42/best.pt` and `retinanet_best.pt` at
`runs/retinanet_curated_seed42/best.pt`. The model code checks the manifest hash
before evaluating either checkpoint.

These are the **trained** checkpoints, not the COCO initialisation weights.
The latter are downloaded by `cetus_models/prepare_cetus.sh`.
