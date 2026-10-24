"""Create contact sheets for human review of audit findings."""

import argparse
import csv
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

COLORS = {"crack": "#f15a3a", "leakage": "#1976d2", "abscission": "#1c9c74",
          "corrosion": "#d07a00", "bulge": "#aa45b8"}
FONT_FILE = "/System/Library/Fonts/Supplemental/Arial.ttf"


def font(size):
    try:
        return ImageFont.truetype(FONT_FILE, size)
    except OSError:
        return ImageFont.load_default()


def annotated_image(root: Path, image_id: str, width: int, height: int) -> Image.Image:
    image = Image.open(root / "JPEGImages" / f"{image_id}.jpg").convert("RGB")
    original_width, original_height = image.size
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    xml = ET.parse(root / "Annotations" / f"{image_id}.xml").getroot()
    for obj in xml.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        b = [float(obj.findtext(f"bndbox/{key}")) for key in ("xmin", "ymin", "xmax", "ymax")]
        x1, y1, x2, y2 = (b[0] * width / original_width, b[1] * height / original_height,
                          b[2] * width / original_width, b[3] * height / original_height)
        draw.rectangle((x1, y1, x2, y2), outline=COLORS.get(name, "red"), width=3)
    return image


def make_pairs(root: Path, source: Path, output: Path, per_page: int,
               prefix: str = "near_duplicates", title: str = "Cross-split near-duplicate review"):
    rows = list(csv.DictReader(source.open(newline="", encoding="utf-8")))
    output.mkdir(parents=True, exist_ok=True)
    f = font(24)
    for page in range(math.ceil(len(rows) / per_page)):
        chunk = rows[page*per_page:(page+1)*per_page]
        canvas = Image.new("RGB", (1120, len(chunk)*355+55), "#f7f9fb")
        draw = ImageDraw.Draw(canvas)
        draw.text((18, 12), f"{title}, page {page+1}", font=f, fill="#163247")
        for index, row in enumerate(chunk):
            y = 55 + index*355
            for side, x in (("A", 18), ("B", 568)):
                image_id = row[f"ImageID_{side}"]
                canvas.paste(annotated_image(root, image_id, 535, 301), (x, y+28))
                draw.text((x, y), f"{image_id}  {row[f'Split_{side}']}  {row[f'GroupID_{side}']}",
                          font=font(20), fill="#163247")
            draw.text((18, y+331),
                      f"dHash={row['DHashDistance']}  thumbnail difference={row['ThumbnailDifference']}",
                      font=font(17), fill="#4d6472")
        canvas.save(output / f"{prefix}_{page+1:02d}.jpg", quality=85)


def make_selection_examples(root: Path, audit_dir: Path, output: Path, per_page: int):
    source = audit_dir / "image_selection_proposal.csv"
    if not source.is_file():
        return
    proposals = [row for row in csv.DictReader(source.open(newline="", encoding="utf-8"))
                 if row["ProposedAction"] == "review_remove_near" and row["CurrentSplit"] == "train"]
    proposals.sort(key=lambda row: float(row["ThumbnailDifference"]))
    if not proposals:
        return
    # Include easy and boundary cases rather than only the closest matches.
    indices = sorted({round(i * (len(proposals)-1) / 14) for i in range(15)})
    rows = []
    for index in indices:
        row = proposals[index]
        rows.append({"ImageID_A": row["ImageID"], "Split_A": row["CurrentSplit"],
                     "GroupID_A": row["GroupID"], "ImageID_B": row["RepresentativeID"],
                     "Split_B": row["CurrentSplit"], "GroupID_B": row["GroupID"],
                     "DHashDistance": "?", "ThumbnailDifference": row["ThumbnailDifference"]})
    sample_path = audit_dir / "near_selection_visual_sample.csv"
    with sample_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    make_pairs(root, sample_path, output, per_page,
               prefix="selection_examples", title="Proposed train-set thinning review")


def make_boxes(root: Path, source: Path, output: Path):
    rows = list(csv.DictReader(source.open(newline="", encoding="utf-8")))
    ids = list(dict.fromkeys(row["ImageID"] for row in rows
                            if row["Severity"] == "blocking" and row["Issue"] == "out_of_bounds_box"))
    output.mkdir(parents=True, exist_ok=True)
    per_page = 12
    for page in range(math.ceil(len(ids)/per_page)):
        chunk = ids[page*per_page:(page+1)*per_page]
        canvas = Image.new("RGB", (1580, 850), "#f7f9fb")
        draw = ImageDraw.Draw(canvas)
        draw.text((18, 12), f"Out-of-bounds box review, page {page+1}",
                  font=font(25), fill="#163247")
        for index, image_id in enumerate(chunk):
            col, row = index % 3, index // 3
            x, y = 18 + col*520, 60 + row*195
            canvas.paste(annotated_image(root, image_id, 500, 165), (x, y+23))
            draw.text((x, y), image_id, font=font(19), fill="#163247")
        canvas.save(output / f"out_of_bounds_{page+1:02d}.jpg", quality=88)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--pairs-per-page", type=int, default=5)
    args = parser.parse_args()
    output = args.audit_dir / "review_sheets"
    make_pairs(args.data_root, args.audit_dir / "high_confidence_cross_split_pairs.csv",
               output, args.pairs_per_page)
    make_selection_examples(args.data_root, args.audit_dir, output, args.pairs_per_page)
    make_boxes(args.data_root, args.audit_dir / "issues.csv", output)
    print(output)


if __name__ == "__main__":
    main()
