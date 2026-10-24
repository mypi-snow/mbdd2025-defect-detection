"""Build a versioned MBDD2025 copy from the conservative curation proposal."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from mbdd_data import CLASSES, EXPECTED_MANIFEST_SHA256, read_manifest

SPLITS = ("train", "val", "test")
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
FEATURES = ("images",) + CLASSES


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def image_number(image_id: str) -> int:
    return int(image_id.removeprefix("Hefei"))


def scene_groups(rows: list[dict], pairs_path: Path, wide_pairs: Path | None = None,
                 manual_scene_links: Path | None = None) -> tuple[dict[str, str], int]:
    parent = {row["GroupID"]: row["GroupID"] for row in rows}
    by_id = {row["ImageID"]: row for row in rows}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(a: str, b: str) -> None:
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    accepted_pairs: set[frozenset[str]] = set()
    for path, threshold in ((pairs_path, 0.03), (wide_pairs, 0.06)):
        if path is None:
            continue
        with path.open(newline="", encoding="utf-8") as stream:
            for pair in csv.DictReader(stream):
                a, b = pair["ImageID_A"], pair["ImageID_B"]
                if (float(pair["ThumbnailDifference"]) <= threshold and
                        abs(image_number(a) - image_number(b)) <= 5):
                    union(by_id[a]["GroupID"], by_id[b]["GroupID"])
                    accepted_pairs.add(frozenset((a, b)))
    if manual_scene_links is not None:
        with manual_scene_links.open(newline="", encoding="utf-8") as stream:
            for link in csv.DictReader(stream):
                if link["Decision"] == "same_scene":
                    union(by_id[link["ImageID_A"]]["GroupID"],
                          by_id[link["ImageID_B"]]["GroupID"])
    roots = sorted({find(group) for group in parent})
    names = {root: f"scene{index:06d}" for index, root in enumerate(roots, 1)}
    return {group: names[find(group)] for group in parent}, len(accepted_pairs)


def annotation_counts(path: Path) -> Counter:
    root = ET.parse(path).getroot()
    return Counter((obj.findtext("name") or "").strip().lower()
                   for obj in root.findall("object"))


def assign_splits(kept: list[dict], source_root: Path,
                  group_to_scene: dict[str, str]) -> tuple[dict[str, str], dict[str, Counter], int]:
    scene_rows = defaultdict(list)
    for row in kept:
        scene_rows[group_to_scene[row["GroupID"]]].append(row)
    scene_features = {}
    assignment = {}
    totals = Counter()
    for scene, members in scene_rows.items():
        counts = Counter({"images": len(members)})
        old_splits = Counter(row["Split"] for row in members)
        assignment[scene] = max(SPLITS, key=lambda split: (old_splits[split], -SPLITS.index(split)))
        for row in members:
            counts.update(annotation_counts(source_root / "Annotations" / (row["ImageID"] + ".xml")))
        scene_features[scene] = counts
        totals.update(counts)

    split_counts = {split: Counter() for split in SPLITS}
    for scene, split in assignment.items():
        split_counts[split].update(scene_features[scene])

    def term(split: str, feature: str, value: int) -> float:
        target = totals[feature] * FRACTIONS[split]
        return (4 if feature == "images" else 1) * ((value-target) / target) ** 2

    moves = 0
    while True:
        best = (0.0, None, None)
        for scene, current in assignment.items():
            features = scene_features[scene]
            for destination in SPLITS:
                if destination == current:
                    continue
                delta = sum(
                    term(current, feature, split_counts[current][feature]-features[feature]) +
                    term(destination, feature, split_counts[destination][feature]+features[feature]) -
                    term(current, feature, split_counts[current][feature]) -
                    term(destination, feature, split_counts[destination][feature])
                    for feature in FEATURES)
                if delta < best[0] - 1e-10:
                    best = (delta, scene, destination)
        if best[1] is None:
            break
        _, scene, destination = best
        current = assignment[scene]
        assignment[scene] = destination
        split_counts[current].subtract(scene_features[scene])
        split_counts[destination].update(scene_features[scene])
        moves += 1
    return assignment, split_counts, moves


def copy_annotation(source: Path, output: Path, width: int, height: int) -> tuple[int, int, str]:
    tree = ET.parse(source)
    root = tree.getroot()
    yolo_lines = []
    changed = 0
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name not in CLASSES:
            raise ValueError(f"Unknown class {name!r} in {source}")
        bnd = obj.find("bndbox")
        if bnd is None:
            raise ValueError(f"Missing box in {source}")
        keys = ("xmin", "ymin", "xmax", "ymax")
        values = [float(bnd.findtext(key, "nan")) for key in keys]
        x1, y1, x2, y2 = values
        clipped = (max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2))
        if not (0 <= clipped[0] < clipped[2] <= width and
                0 <= clipped[1] < clipped[3] <= height):
            raise ValueError(f"Box cannot be clipped to positive area: {source} {values}")
        if tuple(values) != clipped:
            changed += 1
            for key, value in zip(keys, clipped):
                bnd.find(key).text = f"{value:.2f}"
        a, b, c, d = clipped
        yolo_lines.append(f"{CLASSES.index(name)} {(a+c)/(2*width):.8f} "
                          f"{(b+d)/(2*height):.8f} {(c-a)/width:.8f} {(d-b)/height:.8f}")
    if changed:
        tree.write(output, encoding="utf-8", xml_declaration=True)
    else:
        shutil.copy2(source, output)
    return len(yolo_lines), changed, "\n".join(yolo_lines) + ("\n" if yolo_lines else "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--wide-pairs", type=Path)
    parser.add_argument("--manual-scene-links", type=Path)
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).resolve().parents[1] /
                        "curation/audit_inputs/source_split_manifest.csv")
    args = parser.parse_args()
    rows, original_digest = read_manifest(args.manifest)
    if original_digest != EXPECTED_MANIFEST_SHA256:
        raise ValueError("This materializer expects the audited original manifest")
    if args.output_root.exists():
        raise FileExistsError(f"Output already exists: {args.output_root}")
    stage = args.output_root.with_name(args.output_root.name + ".incomplete")
    if stage.exists():
        raise FileExistsError(f"Incomplete staging directory exists: {stage}")
    with args.proposal.open(newline="", encoding="utf-8") as stream:
        proposal = {row["ImageID"]: row for row in csv.DictReader(stream)}
    if {row["ImageID"] for row in rows} != set(proposal):
        raise ValueError("Proposal does not cover the original manifest exactly")
    kept = [row for row in rows if proposal[row["ImageID"]]["ProposedAction"] == "keep"]
    kept_ids = {row["ImageID"] for row in kept}
    for row in rows:
        entry = proposal[row["ImageID"]]
        if entry["ProposedAction"] != "keep" and entry["RepresentativeID"] not in kept_ids:
            raise ValueError(f"Removed image has no retained representative: {row['ImageID']}")
    hashes = [row["SHA256"] for row in kept]
    if len(hashes) != len(set(hashes)):
        raise ValueError("Exact image duplicates remain in the selection")

    group_to_scene, accepted_pairs = scene_groups(
        rows, args.pairs, args.wide_pairs, args.manual_scene_links)
    assignments, split_counts, moves = assign_splits(kept, args.source_root, group_to_scene)
    for folder in ("JPEGImages", "Annotations", "Labels"):
        (stage / folder).mkdir(parents=True)
    output_rows = []
    clipped_boxes = 0
    changed_xml = 0
    empty_images = 0
    for index, row in enumerate(kept, 1):
        image_id = row["ImageID"]
        source_image = args.source_root / "JPEGImages" / f"{image_id}.jpg"
        copied_image = stage / "JPEGImages" / f"{image_id}.jpg"
        shutil.copy2(source_image, copied_image)
        if file_sha256(copied_image) != row["SHA256"]:
            raise ValueError(f"Copied image checksum mismatch: {image_id}")
        source_xml = args.source_root / "Annotations" / f"{image_id}.xml"
        output_xml = stage / "Annotations" / f"{image_id}.xml"
        n_objects, n_clipped, yolo_text = copy_annotation(
            source_xml, output_xml, int(row["Width"]), int(row["Height"]))
        (stage / "Labels" / f"{image_id}.txt").write_text(yolo_text, encoding="utf-8")
        clipped_boxes += n_clipped
        changed_xml += bool(n_clipped)
        empty_images += n_objects == 0
        scene = group_to_scene[row["GroupID"]]
        output_rows.append({
            "ImageID": image_id, "Width": row["Width"], "Height": row["Height"],
            "SHA256": row["SHA256"], "AnnotationSHA256": file_sha256(output_xml),
            "StructuralType": row["StructuralType"], "GroupID": scene,
            "Split": assignments[scene], "OriginalGroupID": row["GroupID"],
            "OriginalSplit": row["Split"],
        })
        if index % 3000 == 0:
            print(f"Copied and verified {index}/{len(kept)} images", flush=True)

    manifest_path = stage / "split_manifest.csv"
    write_csv(manifest_path, output_rows, list(output_rows[0]))
    split_dir = stage / "ImageSets" / "Main"
    split_dir.mkdir(parents=True)
    for split in SPLITS:
        names = [row["ImageID"] for row in output_rows if row["Split"] == split]
        (split_dir / f"{split}.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    (split_dir / "trainval.txt").write_text(
        "\n".join(row["ImageID"] for row in output_rows
                  if row["Split"] in {"train", "val"}) + "\n", encoding="utf-8")
    # The same scene group must occur in exactly one split.
    read_manifest(manifest_path)
    split_by_image = {row["ImageID"]: row["Split"] for row in output_rows}
    scene_by_image = {row["ImageID"]: row["GroupID"] for row in output_rows}
    for path, threshold in ((args.pairs, 0.03), (args.wide_pairs, 0.06)):
        if path is None:
            continue
        with path.open(newline="", encoding="utf-8") as stream:
            for pair in csv.DictReader(stream):
                a, b = pair["ImageID_A"], pair["ImageID_B"]
                if (a in kept_ids and b in kept_ids and
                        float(pair["ThumbnailDifference"]) <= threshold and
                        abs(image_number(a) - image_number(b)) <= 5):
                    if scene_by_image[a] != scene_by_image[b] or split_by_image[a] != split_by_image[b]:
                        raise AssertionError(f"Screened similar pair crosses new splits: {a}, {b}")

    removal_log = []
    for row in rows:
        entry = proposal[row["ImageID"]]
        if entry["ProposedAction"] != "keep":
            removal_log.append({"ImageID": row["ImageID"], "OriginalSplit": row["Split"],
                                "Reason": entry["ProposedAction"],
                                "RepresentativeID": entry["RepresentativeID"],
                                "ThumbnailDifference": entry["ThumbnailDifference"]})
    write_csv(stage / "removed_images.csv", removal_log, list(removal_log[0]))
    review_dir = stage / "curation_review"
    review_dir.mkdir()
    for name in ("image_selection_proposal.csv", "AUDIT_REPORT.md",
                 "cross_split_pair_decisions.csv", "annotation_decisions.csv",
                 "manual_scene_links.csv"):
        source = args.proposal.parent / name
        if source.is_file():
            shutil.copy2(source, review_dir / name)
    sheets = args.proposal.parent / "review_sheets"
    if sheets.is_dir():
        shutil.copytree(sheets, review_dir / "review_sheets")
    summary = {
        "source_images": len(rows), "curated_images": len(kept),
        "removed_images": len(removal_log),
        "removal_reasons": dict(Counter(row["Reason"] for row in removal_log)),
        "scene_groups": len(set(group_to_scene.values())),
        "screened_similarity_pairs_grouped": accepted_pairs,
        "split_balancing_moves": moves,
        "split_counts": {split: dict(split_counts[split]) for split in SPLITS},
        "clipped_out_of_bounds_boxes": clipped_boxes,
        "images_with_modified_xml": changed_xml,
        "images_with_no_labelled_objects": empty_images,
        "source_manifest_sha256": original_digest,
        "curated_manifest_sha256": file_sha256(manifest_path),
    }
    (stage / "curation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (stage / "README.md").write_text(
        "# MBDD2025 curated v1\n\n"
        "This is a copied, reduced derivative of MBDD2025, source: "
        "https://doi.org/10.5281/zenodo.15622584 (CC BY 4.0). "
        "The original download was not changed.\n\n"
        "Images were selected using `image_selection_proposal.csv`: exact copies "
        "were removed and close repeated views within a source group were thinned "
        "only when a retained view had at least as many labels of each class. "
        "Near-repeat decisions are automated and may omit distinct defects; "
        "review `removed_images.csv` and the audit contact sheets before treating "
        "this as final ground truth.\n\n"
        "The 70/15/15 split groups original GroupIDs and adjacent visually "
        "similar frames into a single scene, then balances images and class "
        "counts. It is a new split: all models, including the team's YOLO "
        "baseline, must be retrained from scratch for a fair comparison. "
        "Visual-hash screening does not prove absence of all scene leakage.\n\n"
        "Out-of-frame boxes were clipped to image edges in copied XML, and YOLO "
        "labels were regenerated from the copied XML. No defect class was "
        "relabelled. Boxes with major clipping and ambiguous semantic labels "
        "still need human review. Augment training images only.\n\n"
        "See `curation_summary.json`, `split_manifest.csv`, `removed_images.csv`, "
        "and `curation_review/` for counts and provenance. Run the dataset "
        "audit against this new manifest before model training.\n", encoding="utf-8")
    stage.rename(args.output_root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
