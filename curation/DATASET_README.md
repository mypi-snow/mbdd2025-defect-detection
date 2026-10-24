# MBDD2025 curated v1

This is a copied, reduced derivative of MBDD2025, source: https://doi.org/10.5281/zenodo.15622584 (CC BY 4.0). The original download was not changed.

Images were selected using `image_selection_proposal.csv`: exact copies were removed and close repeated views within a source group were thinned only when a retained view had at least as many labels of each class. Near-repeat decisions are automated and may omit distinct defects; review `removed_images.csv` and the audit contact sheets before treating this as final ground truth.

The 70/15/15 split groups original GroupIDs and adjacent visually similar frames into a single scene, then balances images and class counts. After a wider check, adjacent pairs with thumbnail difference up to 0.06 and two visually confirmed nonadjacent pairs were grouped as well; decisions are in `curation_review/manual_scene_links.csv`. It is a new split: all models, including the team's YOLO baseline, must be retrained from scratch for a fair comparison. Visual-hash screening does not prove absence of all scene leakage.

Out-of-frame boxes were clipped to image edges in copied XML, and YOLO labels were regenerated from the copied XML. No defect class was relabelled. Boxes with major clipping and ambiguous semantic labels still need human review. Augment training images only.

See `curation_summary.json`, `split_manifest.csv`, `removed_images.csv`, `verification/`, and `curation_review/` for counts, checks, and provenance.
`ImageSets/Main/` also contains train, validation, test, and trainval image-ID lists for VOC-style loaders.

Final verification found 12,026 complete image/XML/YOLO triples, 48,163 boxes, no exact-image copies, no boxes outside image bounds, and matching XML/YOLO coordinates. A wider dHash/thumbnail screen retained seven cross-split candidate pairs; all seven were visually reviewed as different scenes (`verification/cross_split_wide_review.csv`). Similarity screening cannot prove that no other related scenes remain. The final manifest SHA-256 is `6f4ab573cd61f1a28120120b10126feda16635e98a0d7db22270dcaba82007c1`.

To rebuild this folder from the original source and audit files, run `cetus_models/materialize_curated_dataset.py` with `--source-root`, `--output-root`, `--proposal`, `--pairs`, `--wide-pairs`, and `--manual-scene-links`. Use a new output path; the script never overwrites an existing folder.
