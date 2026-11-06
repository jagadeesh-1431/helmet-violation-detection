"""Inspect a YOLO-format dataset for label problems. Read-only: it never modifies the dataset.

    python scripts/inspect_dataset.py --root data/raw --out results/dataset_report.json

Also checks for LEAKAGE between splits (the same, or a near-identical, image in train and valid/test),
which makes evaluation scores look better than the model really is.

Checks: malformed lines, class IDs outside the range, coordinates outside [0, 1], zero-size or
tiny boxes, duplicate boxes, polygon (segmentation) lines, label files without images, images
without labels, class balance per split, per-class box-size distribution, class names in
data.yaml versus this project's expected mapping, and helmet-class boxes that are suspiciously
large or that contain several other helmet-class boxes (the "one box around several riders" pattern).

The report is descriptive. A large helmet box is a *flag to look at*, not proof that a label is wrong.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.detector import EXPECTED_CLASSES, HELMET_CLASSES, _norm  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPLITS = ("train", "valid", "val", "test")
LARGE_AREA = 0.08      # fraction of the image
TINY_AREA = 1e-5


def find_image(label: Path) -> Path | None:
    parts = list(label.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "labels":
            for ext in IMG_EXT:
                cand = Path(*parts[:i], "images", *parts[i + 1:-1], label.stem + ext)
                if cand.exists():
                    return cand
                cand = Path(*parts[:i], "images", *parts[i + 1:-1], label.stem + ext.upper())
                if cand.exists():
                    return cand
    return None


def split_of(label: Path, root: Path) -> str:
    for part in label.relative_to(root).parts:
        if part in SPLITS:
            return "valid" if part == "val" else part
    return "unknown"


POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def label_for(img: Path) -> Path:
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            return Path(*parts[:i], "labels", *parts[i + 1:-1], img.stem + ".txt")
    return img.with_suffix(".txt")


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(path: Path):
    """64-bit difference hash: similar-looking images give hashes that differ in only a few bits."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return np.packbits(bits).view(">u8")[0]


def leakage_report(split_images: dict, root: Path, near_threshold: int, max_examples: int) -> dict:
    """Compare images across splits by exact content hash and by perceptual hash."""
    known = {k: v for k, v in split_images.items() if k != "unknown" and v}
    entries = {}           # split -> list of (rel path, md5, dhash)
    unreadable = 0
    for sp, paths in known.items():
        rows = []
        for img in paths:
            try:
                md5 = file_md5(img)
            except OSError:
                unreadable += 1
                continue
            h = dhash(img)
            if h is None:
                unreadable += 1
                continue
            rows.append((str(img.relative_to(root)), md5, h))
        entries[sp] = rows
    by_md5 = defaultdict(list)
    for sp, rows in entries.items():
        for rel, md5, _ in rows:
            by_md5[md5].append((sp, rel))
    exact_cross, exact_within = [], 0
    for items in by_md5.values():
        if len(items) > 1:
            if len({sp for sp, _ in items}) > 1:
                exact_cross.append(items)
            else:
                exact_within += len(items) - 1
    exact_keys = {(a[1], b[1]) for items in exact_cross for a in items for b in items}
    near = []
    splits = sorted(entries)
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            if not entries[a] or not entries[b]:
                continue
            ha = np.array([h for _, _, h in entries[a]], dtype=">u8")
            hb = np.array([h for _, _, h in entries[b]], dtype=">u8")
            small_first = len(ha) <= len(hb)
            probe, pool, pe, pp = (ha, hb, entries[a], entries[b]) if small_first else (hb, ha, entries[b], entries[a])
            for k, h in enumerate(probe):
                dist = POPCOUNT[(pool ^ h).view(np.uint8).reshape(-1, 8)].sum(axis=1)
                for j in np.nonzero(dist <= near_threshold)[0]:
                    pa, pb = pe[k], pp[j]
                    if pa[1] == pb[1] or (pa[0], pb[0]) in exact_keys:
                        continue            # already counted as an exact duplicate
                    near.append({"a": pa[0], "b": pb[0], "hamming": int(dist[j])})
    return {
        "images_checked": sum(len(v) for v in entries.values()),
        "unreadable_images": unreadable,
        "near_duplicate_hamming_threshold": near_threshold,
        "exact_duplicates_across_splits": {"count": len(exact_cross),
                                           "examples": [[f"{sp}: {rel}" for sp, rel in it] for it in exact_cross[:max_examples]]},
        "exact_duplicates_within_a_split": exact_within,
        "near_duplicates_across_splits": {"count": len(near), "examples": near[:max_examples]},
    }


def load_names(root: Path):
    ymls = sorted(root.rglob("data.yaml")) + sorted(root.rglob("*.yaml"))
    for y in ymls:
        try:
            import yaml
            data = yaml.safe_load(y.read_text())
            names = data.get("names") if isinstance(data, dict) else None
            if names:
                if isinstance(names, list):
                    names = dict(enumerate(names))
                return str(y), {int(k): v for k, v in names.items()}
        except Exception:
            continue
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="dataset root (searched recursively for 'labels' folders)")
    ap.add_argument("--out", default="results/dataset_report.json")
    ap.add_argument("--max-examples", type=int, default=10)
    ap.add_argument("--no-leakage", action="store_true", help="skip the (slower) duplicate-image check")
    ap.add_argument("--near-threshold", type=int, default=4,
                    help="max differing bits (of 64) for two images to count as near-duplicates")
    args = ap.parse_args()
    root = Path(args.root)
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory")
        return 2
    labels = sorted(p for p in root.rglob("*.txt") if "labels" in p.parts and p.name.lower() != "classes.txt")
    if not labels:
        print("ERROR: no label .txt files found under a 'labels' folder.")
        return 2

    yaml_path, names = load_names(root)
    n_classes = len(names) if names else len(EXPECTED_CLASSES)
    problems: dict[str, list] = defaultdict(list)
    counts = defaultdict(Counter)                    # split -> class -> boxes
    areas = defaultdict(list)                        # class -> area fractions
    aspects = defaultdict(list)
    images_per_split = Counter()
    empty_labels = Counter()
    helmet_container: list[dict] = []
    split_images: dict[str, list[Path]] = defaultdict(list)

    def note(kind, where, extra=""):
        problems[kind].append(f"{where} {extra}".strip())

    for lab in labels:
        split = split_of(lab, root)
        img = find_image(lab)
        if img is None:
            note("label_without_image", lab.relative_to(root))
        else:
            split_images[split].append(img)
        images_per_split[split] += 1
        rows = []
        seen = set()
        text = lab.read_text(errors="replace").splitlines()
        if not any(t.strip() for t in text):
            empty_labels[split] += 1
        for ln, line in enumerate(text, 1):
            line = line.strip()
            if not line:
                continue
            tok = line.split()
            where = f"{lab.relative_to(root)}:{ln}"
            if len(tok) != 5:
                note("polygon_or_malformed_line" if len(tok) > 5 else "malformed_line", where, f"({len(tok)} fields)")
                continue
            try:
                cid = int(float(tok[0]))
                if float(tok[0]) != cid:
                    raise ValueError
                x, y, w, h = (float(t) for t in tok[1:])
            except ValueError:
                note("malformed_line", where, "(non-numeric)")
                continue
            if not 0 <= cid < n_classes:
                note("class_id_out_of_range", where, f"(id {cid})")
                continue
            if not all(np.isfinite(v) for v in (x, y, w, h)):
                note("non_finite_coordinate", where)
                continue
            if min(x, y, w, h) < 0 or max(x, y, w, h) > 1 or x - w / 2 < -1e-3 or x + w / 2 > 1 + 1e-3 \
                    or y - h / 2 < -1e-3 or y + h / 2 > 1 + 1e-3:
                note("coordinates_outside_image", where, f"({x:.3f},{y:.3f},{w:.3f},{h:.3f})")
            if w <= 0 or h <= 0:
                note("zero_size_box", where)
                continue
            if w * h < TINY_AREA:
                note("tiny_box", where, f"(area {w*h:.2e})")
            if line in seen:
                note("duplicate_box", where)
            seen.add(line)
            counts[split][cid] += 1
            areas[cid].append(w * h)
            aspects[cid].append(max(w / h, h / w))
            rows.append((cid, x, y, w, h))
        # helmet-class boxes that contain several other helmet-class box centres
        hel = [r for r in rows if r[0] in HELMET_CLASSES]
        for r in hel:
            _, x, y, w, h = r
            inside = sum(1 for o in hel if o is not r and abs(o[1] - x) <= w / 2 and abs(o[2] - y) <= h / 2
                         and o[3] * o[4] < 0.5 * w * h)
            if inside >= 2:
                helmet_container.append({"label": str(lab.relative_to(root)), "class": r[0], "contains": inside,
                                         "area": round(w * h, 4)})

    # images that have no label file at all (may be legitimate background images)
    labelled = {i for v in split_images.values() for i in v}
    for img in sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT and "images" in p.parts):
        if img not in labelled and not label_for(img).exists():
            note("image_without_label_file", img.relative_to(root), "(background image or missing label)")
            split_images[split_of(img, root)].append(img)

    # ---------------------------------------------------------------- report
    report: dict = {"root": str(root), "data_yaml": yaml_path, "label_files": len(labels)}
    print(f"Label files: {len(labels)}   data.yaml: {yaml_path or 'not found'}")
    if names:
        report["class_names_in_yaml"] = names
        print("Class names in data.yaml:", names)
        mism = [f"{i}: yaml='{names.get(i)}' expected='{n}'" for i, n in EXPECTED_CLASSES.items()
                if _norm(names.get(i, "")) != _norm(n)]
        report["class_mapping_mismatches"] = mism
        print("Mapping vs project:", "OK" if not mism and len(names) == 4 else f"MISMATCH {mism or '(class count differs)'}")

    print("\nBoxes per split and class")
    per_split = {}
    hdr = "".join(f"{EXPECTED_CLASSES[c]:>15s}" for c in EXPECTED_CLASSES)
    print(f"{'split':10s}{'label files':>12s}{'empty':>8s}{hdr}")
    for sp in sorted(images_per_split):
        row = [counts[sp][c] for c in EXPECTED_CLASSES]
        per_split[sp] = {"label_files": images_per_split[sp], "empty": empty_labels[sp],
                         "boxes": dict(zip(EXPECTED_CLASSES.values(), row))}
        print(f"{sp:10s}{images_per_split[sp]:12d}{empty_labels[sp]:8d}" + "".join(f"{v:15d}" for v in row))
    report["per_split"] = per_split

    print("\nBox size as a fraction of the image area")
    print(f"{'class':15s}{'n':>8s}{'median':>10s}{'p95':>10s}{'max':>10s}{'>8% image':>11s}{'median aspect':>15s}")
    size = {}
    for cid, name in EXPECTED_CLASSES.items():
        a = np.array(areas[cid])
        if len(a):
            size[name] = {"n": int(len(a)), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
                          "max": float(a.max()), "frac_over_8pct": float((a > LARGE_AREA).mean()),
                          "median_aspect": float(np.median(aspects[cid]))}
            s = size[name]
            print(f"{name:15s}{s['n']:8d}{s['median']:10.4f}{s['p95']:10.4f}{s['max']:10.4f}"
                  f"{s['frac_over_8pct']:11.1%}{s['median_aspect']:15.2f}")
        else:
            print(f"{name:15s}{0:8d}")
    report["box_size"] = size
    report["helmet_boxes_containing_multiple_helmet_boxes"] = {
        "count": len(helmet_container), "examples": helmet_container[: args.max_examples]}
    print(f"\nHelmet-class boxes that contain 2+ smaller helmet-class boxes: {len(helmet_container)}")

    print("\nProblems found")
    report["problems"] = {}
    if not problems:
        print("  none of the structural checks failed")
    for kind, items in sorted(problems.items()):
        report["problems"][kind] = {"count": len(items), "examples": items[: args.max_examples]}
        print(f"  {kind}: {len(items)}  e.g. {items[0]}")

    if not args.no_leakage:
        print("\nLeakage between splits (same or near-identical images in different splits)")
        lk = leakage_report(split_images, root, args.near_threshold, args.max_examples)
        report["leakage"] = lk
        print(f"  images checked: {lk['images_checked']}  unreadable: {lk['unreadable_images']}")
        print(f"  exact duplicates across splits: {lk['exact_duplicates_across_splits']['count']}")
        print(f"  near-duplicates across splits (<= {args.near_threshold}/64 bits): {lk['near_duplicates_across_splits']['count']}")
        print(f"  exact duplicates inside one split: {lk['exact_duplicates_within_a_split']}")
        if lk["exact_duplicates_across_splits"]["count"] or lk["near_duplicates_across_splits"]["count"]:
            print("  WARNING: evaluation images that also appear in training make mAP look better than it is.")
            print("  Near-duplicate pairs are candidates to LOOK at (video frames of one scene often are); they are not proof of a mistake.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(f"\nReport written to {args.out}")
    print("Next: open a few flagged images with their labels drawn on them before concluding anything about label quality.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
