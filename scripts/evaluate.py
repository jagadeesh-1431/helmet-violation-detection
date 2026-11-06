"""Evaluate the trained model on labelled data and write a reproducible report.

    python scripts/evaluate.py --data data/raw/data.yaml --split test --model models/best.pt

Writes results/evaluation.json and reports/model_evaluation.md. Every number in them is measured on the
split you name; nothing is copied from earlier runs.

Two validation passes are made, because the two kinds of numbers need different confidence thresholds:
  1. conf=0.001  -> precision/recall curve metrics: mAP@50 and mAP@50-95 (the standard detection scores).
  2. conf=--cm-conf (default 0.25, the web app's default) -> confusion matrix, and per-class
     precision and recall AT THAT THRESHOLD: what you actually get when you use the app at 0.25.

Requirements: data.yaml must point at folders that exist on THIS machine (edit its 'path:' line if it
still refers to Colab paths). Use a split the model was NOT trained or tuned on. If the dataset has no
'test' split, use --split val, and understand those numbers are optimistic (best.pt was picked on them).
Raw detection counts on a video are not accuracy. These metrics, computed against human labels, are.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.detector import EXPECTED_CLASSES, validate_class_names  # noqa: E402

NAMES = [EXPECTED_CLASSES[i] for i in sorted(EXPECTED_CLASSES)]


def _ratio(num: float, den: float):
    return round(num / den, 4) if den else None


def operating_point(matrix, names=NAMES) -> dict:
    """Per-class TP/FP/FN, precision and recall from a confusion matrix indexed [predicted, actual]
    with a trailing background row/column (the Ultralytics layout)."""
    n = len(names)
    out = {}
    for c, name in enumerate(names):
        tp = float(matrix[c][c])
        fp = float(sum(matrix[c][j] for j in range(n + 1))) - tp      # predicted c, actually another class or background
        fn = float(sum(matrix[i][c] for i in range(n + 1))) - tp      # actually c, predicted another class or missed
        out[name] = {"tp": int(tp), "fp": int(fp), "fn": int(fn),
                     "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn)}
    return out


def _fmt(v, digits=3):
    return "n/a" if v is None else f"{v:.{digits}f}"


def build_markdown(rep: dict) -> str:
    m, d, st, hw = rep["model"], rep["dataset"], rep["settings"], rep["hardware"]
    L = []
    L.append("# Model evaluation\n")
    L.append(f"Generated {rep['generated_at']} by `scripts/evaluate.py`. All numbers below were measured on the split named here.\n")
    L.append("## What was evaluated\n")
    L.append(f"| | |\n|---|---|\n| Weights | `{m['file']}` ({m['size_mb']} MB, sha256 `{m['sha256_12']}...`) |")
    L.append(f"| Dataset config | `{d['data_yaml']}` |\n| Split | `{d['split']}` |")
    L.append(f"| Images in split | {d['images'] if d['images'] is not None else 'not counted'} |")
    L.append(f"| Image size | {st['imgsz']} px |\n")
    if d["split"] != "test":
        L.append(f"> **Caution:** the `{d['split']}` split is normally the one used to choose `best.pt` during training, so these scores are optimistic. Evaluate on a held-out `test` split before quoting them.\n")
    L.append("## Per-class detection scores\n")
    L.append(f"mAP uses confidence 0.001 and IoU 0.6. Precision and recall in this table are at the confidence that maximises F1 for each class (the Ultralytics convention), not at the app's default.\n")
    L.append("| Class | Instances | Precision | Recall | mAP@50 | mAP@50-95 |\n|---|---|---|---|---|---|")
    for name in NAMES:
        r = rep["per_class"].get(name, {})
        L.append(f"| {name} | {r.get('instances', 'n/a')} | {_fmt(r.get('precision'))} | {_fmt(r.get('recall'))} | {_fmt(r.get('map50'))} | {_fmt(r.get('map50_95'))} |")
    o = rep["overall"]
    L.append(f"| **All** | {sum(v.get('instances') or 0 for v in rep['per_class'].values())} | {_fmt(o['precision'])} | {_fmt(o['recall'])} | {_fmt(o['map50'])} | {_fmt(o['map50_95'])} |\n")
    op = rep["operating_point"]
    if op is None:
        L.append("## Threshold-based counts and confusion matrix\n")
        L.append("Not available: the confusion matrix could not be computed or failed its consistency check, so no TP/FP/FN counts are shown. Do not infer them from the table above.\n")
        return _finish(L, rep)
    L.append(f"## At the app's default threshold (confidence {op['conf']})\n")
    L.append("Counts come from matching each prediction to a ground-truth box (IoU 0.45, same class). **TP** correct, **FP** wrong or no object there, **FN** real object missed.\n")
    L.append("| Class | TP | FP | FN | Precision | Recall |\n|---|---|---|---|---|---|")
    for name in NAMES:
        r = op["per_class"][name]
        L.append(f"| {name} | {r['tp']} | {r['fp']} | {r['fn']} | {_fmt(r['precision'])} | {_fmt(r['recall'])} |")
    L.append("")
    cm = rep["confusion_matrix"]
    L.append(f"## Confusion matrix (confidence {op['conf']})\n")
    L.append("Rows are what the model **predicted**, columns are what was **actually there**. `background` means no object: a prediction in the background column is a false alarm, an actual object in the background row was missed.\n")
    L.append("| predicted \\ actual | " + " | ".join(cm["labels"]) + " |\n|---|" + "---|" * len(cm["labels"]))
    for lab, row in zip(cm["labels"], cm["matrix"]):
        L.append(f"| **{lab}** | " + " | ".join(str(int(v)) for v in row) + " |")
    L.append("")
    return _finish(L, rep)


def _finish(L: list, rep: dict) -> str:
    st, hw = rep["settings"], rep["hardware"]
    sp = rep["speed_ms_per_image"]
    L.append("## Speed and hardware\n")
    L.append(f"Per image on this machine: preprocess {sp['preprocess']:.1f} ms, inference {sp['inference']:.1f} ms, postprocess {sp['postprocess']:.1f} ms.\n")
    L.append(f"| | |\n|---|---|\n| Device | {st['device']} |\n| CPU | {hw['processor']} ({hw['cpu_count']} logical cores) |\n| OS | {hw['platform']} |")
    L.append(f"| Python / PyTorch / Ultralytics | {hw['python']} / {hw['torch']} / {hw['ultralytics']} |\n")
    L.append("## Limitations of this report\n")
    L.append("- Scores describe this dataset and split only. They do not show accuracy on other cameras, lighting, weather or road scenes.")
    L.append("- Small classes (few instances) give unstable scores; check the Instances column before trusting a number.")
    L.append("- `TripleRiding` is a composite class. A correct box does not by itself establish the number of riders.")
    L.append("- Representative false positives and false negatives are not generated here. Review them with `scripts/visualize_dataset.py` and the web app's review panel.")
    L.append("- A prediction counted correct here is a match to a label under the stated IoU and class rules, not a confirmed real-world violation.\n")
    return "\n".join(L)


def _count_images(data_yaml: Path, split: str):
    try:
        import yaml
        cfg = yaml.safe_load(data_yaml.read_text())
        base = Path(cfg.get("path", data_yaml.parent))
        if not base.is_absolute() or not base.exists():
            base = data_yaml.parent
        entry = cfg.get({"val": "val", "test": "test", "train": "train"}[split])
        if not isinstance(entry, str):
            return None
        d = base / entry
        if not d.is_dir():
            return None
        return sum(1 for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="path to the dataset data.yaml")
    ap.add_argument("--model", default="models/best.pt")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cm-conf", type=float, default=0.25, help="confidence for the confusion matrix and operating-point metrics")
    ap.add_argument("--out", default="results/evaluation.json")
    ap.add_argument("--report", default="reports/model_evaluation.md")
    args = ap.parse_args()

    import torch
    import ultralytics
    from ultralytics import YOLO

    model_path, data_path = Path(args.model), Path(args.data)
    if not model_path.is_file():
        print(f"ERROR: model not found: {model_path}"); return 2
    if not data_path.is_file():
        print(f"ERROR: data.yaml not found: {data_path}"); return 2
    model = YOLO(str(model_path))
    validate_class_names(model.names)
    plots_dir = Path(args.report).parent / "eval_plots"

    common = dict(data=str(data_path), split=args.split, imgsz=args.imgsz, device=args.device, plots=False, verbose=False)
    print("Pass 1/2: mAP (confidence 0.001)...")
    with tempfile.TemporaryDirectory() as scratch:      # keeps stray 'runs/' folders out of your project
        m1 = model.val(conf=0.001, iou=0.6, project=scratch, name="map", exist_ok=True, **common)
    print(f"Pass 2/2: confusion matrix at confidence {args.cm_conf} (plots are saved to {plots_dir})...")
    # Ultralytics only fills the confusion matrix when plots=True, so pass 2 must run with plots on.
    m2 = model.val(conf=args.cm_conf, iou=0.45, **{**common, "plots": True},
                   project=str(plots_dir.parent), name=plots_dir.name, exist_ok=True)

    box = m1.box
    nt_arr = getattr(m1, "nt_per_class", None)
    nt = [] if nt_arr is None else [int(x) for x in nt_arr]
    per_class = {}
    for k, cid in enumerate(box.ap_class_index):
        name = EXPECTED_CLASSES[int(cid)]
        per_class[name] = {"instances": int(nt[int(cid)]) if int(cid) < len(nt) else None,
                           "precision": float(box.p[k]), "recall": float(box.r[k]),
                           "map50": float(box.ap50[k]), "map50_95": float(box.ap[k])}
    matrix = [[float(v) for v in row] for row in m2.confusion_matrix.matrix]
    n = len(NAMES)
    cm_ok = sum(sum(r) for r in matrix) > 0
    if cm_ok and nt:
        # every labelled object appears in exactly one column (as TP, a wrong class, or a miss)
        col_totals = [int(round(sum(matrix[i][c] for i in range(n + 1)))) for c in range(n)]
        cm_ok = col_totals == [nt[c] if c < len(nt) else 0 for c in range(n)]
        if not cm_ok:
            print(f"WARNING: confusion-matrix column totals {col_totals} do not match the labelled instances {nt}; "
                  "the confusion matrix and the threshold-based counts are omitted from the report.")
    elif not cm_ok:
        print("WARNING: the confusion matrix is empty, so it and the threshold-based counts are omitted from the report.")
    sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    report = {
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model": {"file": model_path.name, "size_mb": round(model_path.stat().st_size / 1e6, 1), "sha256_12": sha[:12]},
        "dataset": {"data_yaml": str(data_path), "split": args.split, "images": _count_images(data_path, args.split)},
        "settings": {"imgsz": args.imgsz, "device": args.device, "map_conf": 0.001, "map_iou": 0.6,
                     "cm_conf": args.cm_conf, "cm_iou": 0.45},
        "overall": {"precision": float(box.mp), "recall": float(box.mr), "map50": float(box.map50), "map50_95": float(box.map)},
        "per_class": per_class,
        "operating_point": {"conf": args.cm_conf, "per_class": operating_point(matrix)} if cm_ok else None,
        "confusion_matrix": {"orientation": "rows = predicted, columns = actual; last = background",
                             "labels": NAMES + ["background"], "matrix": matrix} if cm_ok else None,
        "speed_ms_per_image": {k: float(v) for k, v in m1.speed.items() if k in ("preprocess", "inference", "postprocess")},
        "hardware": {"processor": platform.processor() or "unknown", "cpu_count": os.cpu_count(),
                     "platform": platform.platform(), "python": platform.python_version(),
                     "torch": torch.__version__, "ultralytics": ultralytics.__version__},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(build_markdown(report), encoding="utf-8")

    print(f"\nSplit: {args.split}   images: {report['dataset']['images']}")
    print(f"{'class':15s}{'inst':>6s}{'prec':>8s}{'recall':>8s}{'mAP50':>8s}{'mAP50-95':>10s}")
    for name in NAMES:
        r = per_class.get(name)
        if r:
            print(f"{name:15s}{str(r['instances']):>6s}{r['precision']:8.3f}{r['recall']:8.3f}{r['map50']:8.3f}{r['map50_95']:10.3f}")
    o = report["overall"]
    print(f"{'all':15s}{'':>6s}{o['precision']:8.3f}{o['recall']:8.3f}{o['map50']:8.3f}{o['map50_95']:10.3f}")
    if report["operating_point"]:
        print(f"\nAt confidence {args.cm_conf}:")
        for name, r in report["operating_point"]["per_class"].items():
            print(f"  {name:15s} TP {r['tp']:4d}  FP {r['fp']:4d}  FN {r['fn']:4d}  precision {_fmt(r['precision'])}  recall {_fmt(r['recall'])}")
    print(f"\nWritten: {args.out}\n         {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
