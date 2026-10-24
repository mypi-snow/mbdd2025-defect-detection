# MBDD2025 data-quality and curation audit

**Status:** reviewable proposal, not a final cleaned dataset. The source at
`/path/to/original/MBDD2025/` was read only. No original image or label was
deleted or silently relabelled. This audit used the teammate's 14,471-image
split manifest (SHA-256 `1ad8ec9c77ae84e64531490df1843e075ed0de2bd891f9e847e93821e9482742`).

## What the audit found

| Finding | Count | Consequence |
| --- | ---: | --- |
| Images / labelled defect boxes | 14,471 / 57,613 | Five classes; the manifest has train 10,066, validation 2,187, test 2,218. |
| Exact-image duplicate groups / extra copies | 166 / 169 | Redundant examples can overweight a scene. All exact copies are within their current split. One group has box coordinates differing by at most one pixel. |
| Out-of-image bounding boxes | 116 | These must be corrected consistently in XML and YOLO labels before final training/evaluation. For 29 images, clipping would remove more than 10% of at least one box; those need visual review first. |
| Images with no labelled objects | 8 | Could be useful background negatives or missing annotations; inspect before deciding. |
| Cross-split visual-similarity candidates | 695 | dHash distance at most 4 and mean thumbnail difference at most 0.06. This is a **candidate list**, not 695 confirmed leaks. |
| Priority cross-split review pairs | 110 | Thumbnail difference at most 0.015 and numeric image-ID gap at most 5. The first review sheet visibly contains nearly identical views on opposite sides of a split. Other pairs still require review. |

All XML and YOLO annotation rows agree numerically and by class. That is a
format-consistency check, **not** evidence that the depicted defect class is
correct: the YOLO files appear to derive from the XML labels. The audit also
found no missing files or image-size mismatches. `StructuralType` is `unknown`
throughout the supplied manifest, so a structure-type split cannot be claimed
without independently verified metadata.

## Conservative image-selection proposal

`image_selection_proposal.csv` has one row per original image. It suggests
retaining **12,026** images and reviewing **2,445** possible removals:

| Current split | Keep | Exact-copy removal proposal | Near-repeat removal for review |
| --- | ---: | ---: | ---: |
| Train | 8,745 | 131 | 1,190 |
| Validation | 1,621 | 10 | 556 |
| Test | 1,660 | 28 (including one with one-pixel annotation jitter) | 530 |

An exact-copy removal is proposed when the image bytes and annotations match,
or when identical image bytes have box coordinates differing by no more than
one pixel. A near-repeat is proposed only if the images share the
original `GroupID` and split, their numeric IDs are within five, thumbnail
difference is at most 0.005, and a directly matched retained image has at
least as many labelled objects of every class. This last rule preserves class
coverage, but it cannot prove that all individual defects are represented.
For example, two nearly identical views may contain different valid defects or
one may have incomplete annotations. **Review the near-repeat decisions before
applying them.** The proposal does not remove or move any file.

The current grouped split prevents the *same GroupID* appearing in multiple
splits, but the review sheets show adjacent, visually near-identical views
with different GroupIDs in train/validation/test. Accordingly, this proposal
is for sizing and review only; **do not treat it as a cleaned training split**.
Visual-hash screening also misses some crops or viewpoint changes and can
mistake featureless walls for matches. The 110 priority pairs are a starting
queue, not a complete guarantee against scene overlap.

## Decisions before model training

1. Review `review_sheets/near_duplicates_*.jpg` and record `same_scene`,
   `different_scene`, or `uncertain` in `cross_split_pair_decisions.csv`. Check
   wider capture sequences around a confirmed pair, not only the two photos.
   Keep all images from one physical wall/inspection scene in **one split**.
2. Review `review_sheets/selection_examples_*.jpg` and then the proposed
   near-repeat rows in `image_selection_proposal.csv`. Retain distinct lighting,
   distance, viewpoint, rare defects, or clearer annotations even if the
   backgrounds look similar. Prefer a representative with sound labels over
   simply the highest box count.
3. Review `review_sheets/out_of_bounds_*.jpg` and mark the 37 images in
   `annotation_decisions.csv` (29 severe out-of-bounds and eight with no
   objects). For each affected box, decide whether to clip, redraw, or exclude
   it. Correct the XML and regenerated YOLO labels together in a **new copy**
   of the dataset. Mild out-of-bounds boxes can be clipped mechanically after
   checking a sample; record that rule in the experiment log.
4. Use `review_queue.csv` to inspect unusually large/small boxes and a seeded,
   stratified sample from each class. Have two people check semantic labels,
   especially visually ambiguous leakage, corrosion, and abscission. Log each
   correction with the original and revised class/box.
5. Rebuild a **versioned, scene-grouped** train/validation/test manifest after
   those decisions. Apply thinning within the resulting split and freeze the
   manifest plus annotation hashes. Check zero exact duplicates and zero
   *confirmed* same-scene pairs across splits. Any unresolved cross-split
   similarity should be reported as a limitation.
6. Rerun the tiny-YOLOv4 baseline and all alternative models on this same
   curated manifest. Results from the old and new splits must be labelled
   separately. Select preprocessing, augmentations and hyperparameters using
   training/validation only; score the held-out test once after choices are
   fixed. Apply augmentation to **training images only**, with boxes transformed
   and clipped consistently.

## Files and reproducibility

- `summary.json`, `issues.csv`, `exact_duplicates.csv`: structural and box audit.
- `all_near_duplicate_candidates.csv`: within- and cross-split screening pairs
  with thumbnail difference at most 0.03. Many are redundant pairwise links;
  it is **not** a count of independent scenes or removals.
- `cross_split_near_duplicates.csv`: wider cross-split candidate queue (0.06).
- `high_confidence_cross_split_pairs.csv`, `cross_split_pair_decisions.csv`:
  priority queue and blank human-decision fields.
- `image_selection_proposal.csv`, `curation_summary.json`: conservative image
  selection proposal and counts; blank human-decision fields.
- `annotation_decisions.csv`, `review_queue.csv`: label/box review queues.
- `review_sheets/`: box-overlay contact sheets for visual decisions.

Reproduce with Python 3.10+, Pillow and NumPy:

```bash
python cetus_models/audit_dataset.py --data-root MBDD2025 --output-dir dataset_audit
python cetus_models/audit_near_duplicates.py --data-root MBDD2025 --output dataset_audit/cross_split_near_duplicates.csv
python cetus_models/audit_near_duplicates.py --data-root MBDD2025 --output dataset_audit/all_near_duplicate_candidates.csv --include-same-split --max-thumbnail-difference 0.03
python cetus_models/propose_curation.py --data-root MBDD2025 --pairs dataset_audit/all_near_duplicate_candidates.csv --output-dir dataset_audit
python cetus_models/make_review_sheets.py --data-root MBDD2025 --audit-dir dataset_audit
```

## Materialized curated copy

The separate `MBDD2025_curated_v1/` folder was subsequently built from this
proposal. It has 12,026 images (train 8,418; validation 1,804; test 1,804),
48,163 boxes, zero exact-image duplicates, and zero out-of-bounds boxes after
clipping. The wider dHash/thumbnail screen (distance <=4; difference <=0.06)
retains seven cross-split candidate pairs, all visually reviewed as different
scenes. This is a screened result, not proof that all physical scenes are
distinct. The copied folder includes
its own versioned split manifest, removal log, README, and verification audit.
It is a usable **provisional** dataset: the automated near-repeat removals and
ambiguous defect labels still warrant team review. No source file in the
original download was changed.

Rebuild the copied dataset (use a new output name if the folder already exists):

```bash
python cetus_models/materialize_curated_dataset.py \
  --source-root MBDD2025 --output-root MBDD2025_curated_v1 \
  --proposal dataset_audit/image_selection_proposal.csv \
  --pairs dataset_audit/all_near_duplicate_candidates.csv \
  --wide-pairs dataset_audit/all_near_duplicate_candidates_wide.csv \
  --manual-scene-links dataset_audit/manual_scene_links.csv
```

Dataset source: Zha, Yao, Zhang & Ma (2025), *MBDD2025: A Deep
Learning-Oriented Dataset for Diverse Building Defect Detection from UAV
Images*, Zenodo, [doi:10.5281/zenodo.15622584](https://doi.org/10.5281/zenodo.15622584).
