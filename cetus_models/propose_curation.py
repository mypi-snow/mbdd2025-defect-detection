"""Create a conservative, reviewable image-selection proposal; never delete data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from mbdd_data import EXPECTED_MANIFEST_SHA256, read_manifest


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def image_number(image_id: str) -> int:
    return int(image_id.removeprefix("Hefei"))


def annotations(path: Path) -> tuple[Counter, tuple]:
    objects = []
    for obj in ET.parse(path).getroot().findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        box = tuple(round(float(obj.findtext(f"bndbox/{key}")), 2)
                    for key in ("xmin", "ymin", "xmax", "ymax"))
        objects.append((name, box))
    return Counter(name for name, _ in objects), tuple(sorted(objects))


def dominates(a: Counter, b: Counter) -> bool:
    """A preserves at least the labelled class counts present in B."""
    return all(a[name] >= count for name, count in b.items())


def tiny_coordinate_jitter(signatures: list[tuple]) -> bool:
    reference = signatures[0]
    for other in signatures[1:]:
        if len(other) != len(reference):
            return False
        if any(name_a != name_b or any(abs(a-b) > 1.0 for a, b in zip(box_a, box_b))
               for (name_a, box_a), (name_b, box_b) in zip(reference, other)):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).resolve().parents[1] /
                        "curation/audit_inputs/source_split_manifest.csv")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-difference", type=float, default=0.005)
    parser.add_argument("--max-id-gap", type=int, default=5)
    args = parser.parse_args()
    rows, digest = read_manifest(args.manifest)
    if digest != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Manifest differs from the original baseline")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_id = {row["ImageID"]: row for row in rows}
    labels = {}
    signatures = {}
    class_total = Counter()
    for row in rows:
        image_id = row["ImageID"]
        labels[image_id], signatures[image_id] = annotations(
            args.data_root / "Annotations" / f"{image_id}.xml")
        class_total.update(labels[image_id])

    def rank(image_id: str):
        counts = labels[image_id]
        rare_score = sum(count / math.sqrt(class_total[name])
                         for name, count in counts.items())
        return (-len(counts), -rare_score, -sum(counts.values()), image_number(image_id))

    action = {image_id: "keep" for image_id in by_id}
    representative = {image_id: image_id for image_id in by_id}
    reason = {image_id: "" for image_id in by_id}
    difference = {image_id: "" for image_id in by_id}
    exact_groups = defaultdict(list)
    for row in rows:
        exact_groups[row["SHA256"]].append(row["ImageID"])
    inconsistent_exact = []
    inconsistent_ids = set()
    for members in exact_groups.values():
        if len(members) < 2:
            continue
        if len({by_id[image_id]["Split"] for image_id in members}) != 1:
            raise ValueError("An exact duplicate crosses the existing splits")
        annotation_jitter = len({signatures[image_id] for image_id in members}) > 1
        if annotation_jitter and not tiny_coordinate_jitter([signatures[image_id] for image_id in members]):
            inconsistent_exact.append(members)
            inconsistent_ids.update(members)
            continue
        kept = min(members, key=rank)
        for image_id in members:
            if image_id != kept:
                action[image_id] = ("propose_remove_exact_jitter" if annotation_jitter
                                    else "propose_remove_exact")
                representative[image_id] = kept
                reason[image_id] = ("same_image_bytes_boxes_differ_at_most_1px" if annotation_jitter
                                    else "same_image_bytes_and_annotations")

    adjacency = defaultdict(dict)
    with args.pairs.open(newline="", encoding="utf-8") as stream:
        for pair in csv.DictReader(stream):
            a, b = pair["ImageID_A"], pair["ImageID_B"]
            if (pair["Split_A"] != pair["Split_B"] or
                    pair["GroupID_A"] != pair["GroupID_B"] or
                    abs(image_number(a) - image_number(b)) > args.max_id_gap):
                continue
            delta = float(pair["ThumbnailDifference"])
            if delta > args.max_difference:
                continue
            adjacency[a][b] = delta
            adjacency[b][a] = delta

    retained = set()
    for image_id in sorted(by_id, key=rank):
        if action[image_id] != "keep":
            continue
        if image_id in inconsistent_ids:
            retained.add(image_id)
            continue
        candidates = [other for other in adjacency[image_id]
                      if other in retained and other not in inconsistent_ids
                      and dominates(labels[other], labels[image_id])]
        if candidates:
            kept = min(candidates, key=lambda other: (adjacency[image_id][other], rank(other)))
            action[image_id] = "review_remove_near"
            representative[image_id] = kept
            reason[image_id] = "same_original_group_and_near_identical_view"
            difference[image_id] = f"{adjacency[image_id][kept]:.6f}"
        else:
            retained.add(image_id)

    for image_id in by_id:
        if action[image_id].startswith("propose_remove_exact"):
            kept = representative[image_id]
            representative[image_id] = representative[kept]

    proposal = []
    for row in rows:
        image_id = row["ImageID"]
        proposal.append({
            "ImageID": image_id, "CurrentSplit": row["Split"],
            "GroupID": row["GroupID"], "ProposedAction": action[image_id],
            "RepresentativeID": representative[image_id],
            "Reason": reason[image_id], "ThumbnailDifference": difference[image_id],
            "AnnotationCount": sum(labels[image_id].values()),
            "Classes": ";".join(sorted(labels[image_id])),
            "ReviewerDecision": "", "Reviewer": "", "Notes": "",
        })
    write_csv(args.output_dir / "image_selection_proposal.csv", proposal,
              list(proposal[0]))

    pair_reviews = []
    high_confidence_path = args.output_dir / "high_confidence_cross_split_pairs.csv"
    if high_confidence_path.is_file():
        with high_confidence_path.open(newline="", encoding="utf-8") as stream:
            for pair in csv.DictReader(stream):
                pair_reviews.append({**pair, "SameSceneDecision": "",
                                     "Reviewer": "", "Notes": ""})
        write_csv(args.output_dir / "cross_split_pair_decisions.csv", pair_reviews,
                  list(pair_reviews[0]))

    issue_reviews = []
    issues_path = args.output_dir / "issues.csv"
    if issues_path.is_file():
        with issues_path.open(newline="", encoding="utf-8") as stream:
            seen = set()
            for issue in csv.DictReader(stream):
                if issue["Severity"] == "blocking" or issue["Issue"] == "no_objects":
                    key = (issue["ImageID"], issue["Issue"])
                    if key not in seen:
                        issue_reviews.append({"ImageID": issue["ImageID"],
                                              "Split": issue["Split"], "Issue": issue["Issue"],
                                              "ReviewerDecision": "", "Reviewer": "", "Notes": ""})
                        seen.add(key)
        write_csv(args.output_dir / "annotation_decisions.csv", issue_reviews,
                  ["ImageID", "Split", "Issue", "ReviewerDecision", "Reviewer", "Notes"])

    summary = {
        "input_manifest_sha256": digest,
        "selection_scope": "proposal_on_existing_split; final splits require scene review and rebuilding",
        "rules": {"same_group_only_for_near_duplicates": True,
                  "max_thumbnail_difference": args.max_difference,
                  "max_numeric_id_gap": args.max_id_gap,
                  "class_count_coverage_required": True},
        "actions_by_split": {split: dict(Counter(item["ProposedAction"] for item in proposal
                                               if item["CurrentSplit"] == split))
                             for split in ("train", "val", "test")},
        "inconsistent_exact_duplicate_groups_for_review": len(inconsistent_exact),
        "cross_split_priority_pairs_for_review": len(pair_reviews),
        "annotation_issue_images_for_review": len(issue_reviews),
    }
    (args.output_dir / "curation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
