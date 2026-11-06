"""Draw the ground-truth labels onto a sample of dataset images and save them as one contact sheet.

    python scripts/visualize_dataset.py --root data/raw --split train --class WithoutHelmet --n 24
    python scripts/visualize_dataset.py --root data/raw --class WithoutHelmet --min-area 0.08

Use --min-area to pull out images that contain a LARGE box of the chosen class (as a fraction of the
image). That is the quickest way to see whether the training labels really contain whole-scooter
"WithoutHelmet" boxes. Read-only: the dataset is never modified.

A large box is not automatically a wrong label. Look at the picture and decide whether it matches what
the class is supposed to mean before changing anything.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.detector import CLASS_COLORS_BGR, EXPECTED_CLASSES  # noqa: E402
from inspect_dataset import find_image, split_of  # noqa: E402

LARGE = 0.08


def read_boxes(label: Path):
    boxes = []
    for line in label.read_text(errors="replace").splitlines():
        tok = line.split()
        if len(tok) != 5:
            continue
        try:
            cid = int(float(tok[0]))
            x, y, w, h = (float(t) for t in tok[1:])
        except ValueError:
            continue
        if cid in EXPECTED_CLASSES and w > 0 and h > 0:
            boxes.append((cid, x, y, w, h))
    return boxes


def class_id(value: str | None):
    if value is None:
        return None
    if value.isdigit():
        return int(value)
    norm = value.lower().replace("_", "").replace("-", "")
    for cid, name in EXPECTED_CLASSES.items():
        if name.lower() == norm:
            return cid
    raise SystemExit(f"Unknown class '{value}'. Use one of: {', '.join(EXPECTED_CLASSES.values())} or 0-3.")


def draw(img: np.ndarray, boxes, highlight: int | None) -> np.ndarray:
    h, w = img.shape[:2]
    for cid, x, y, bw, bh in boxes:
        x1, y1, x2, y2 = int((x - bw / 2) * w), int((y - bh / 2) * h), int((x + bw / 2) * w), int((y + bh / 2) * h)
        big = bw * bh > LARGE
        color = CLASS_COLORS_BGR[cid]
        thick = 4 if (highlight is not None and cid == highlight) else 2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        label = f"{EXPECTED_CLASSES[cid]} {bw * bh:.1%}" + (" LARGE" if big else "")
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(y1 - 4, th + 4)
        cv2.rectangle(img, (x1, ty - th - 3), (x1 + tw + 4, ty + base - 1), color, -1)
        cv2.putText(img, label, (x1 + 2, ty - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="any", choices=["any", "train", "valid", "test"])
    ap.add_argument("--class", dest="cls", help="only images containing this class (name or id 0-3)")
    ap.add_argument("--min-area", type=float, default=0.0,
                    help="with --class: only images where a box of that class covers at least this fraction of the image")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--cell-width", type=int, default=420)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports/contact_sheet.jpg")
    args = ap.parse_args()

    root = Path(args.root)
    want = class_id(args.cls)
    candidates = []
    for lab in sorted(root.rglob("*.txt")):
        if "labels" not in lab.parts:
            continue
        if args.split != "any" and split_of(lab, root) != args.split:
            continue
        img = find_image(lab)
        if img is None:
            continue
        boxes = read_boxes(lab)
        if want is not None:
            if not any(b[0] == want and b[3] * b[4] >= args.min_area for b in boxes):
                continue
        candidates.append((img, lab, boxes))
    if not candidates:
        print("No images matched. Check --root, --split, --class and --min-area.")
        return 1
    random.Random(args.seed).shuffle(candidates)
    chosen = candidates[: args.n]

    cells = []
    for img_path, _, boxes in chosen:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        scale = args.cell_width / img.shape[1]
        img = cv2.resize(img, (args.cell_width, max(int(img.shape[0] * scale), 1)))
        draw(img, boxes, want)
        cv2.putText(img, img_path.name[:40], (4, img.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, img_path.name[:40], (4, img.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        cells.append(img)
    if not cells:
        print("The matching images could not be read.")
        return 1
    cell_h = max(c.shape[0] for c in cells)
    rows = []
    for i in range(0, len(cells), args.cols):
        row = cells[i:i + args.cols]
        row = [cv2.copyMakeBorder(c, 0, cell_h - c.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30)) for c in row]
        while len(row) < args.cols:
            row.append(np.full((cell_h, args.cell_width, 3), 30, np.uint8))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    listing = out.with_suffix(".txt")
    listing.write_text("\n".join(f"{i + 1}\t{img.relative_to(root)}\t{lab.relative_to(root)}"
                                 for i, (img, lab, _) in enumerate(chosen)) + "\n", encoding="utf-8")
    print(f"{len(candidates)} matching images; showing {len(cells)}.")
    print(f"Contact sheet: {out}  ({sheet.shape[1]}x{sheet.shape[0]})")
    print(f"Image list (in order, left to right, top to bottom): {listing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
