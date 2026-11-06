"""Audit the trained model: class names, and (optionally) behaviour on real footage.

    python scripts/audit_model.py --model models/best.pt --video videos/sample.mp4 --frames 40

What it reports (all measured, nothing assumed):
  * the class names stored in the weights and whether they match the 4-class mapping
  * on sampled video frames: detections per class, confidence summary, and box SIZE statistics
    for WithHelmet / WithoutHelmet. This is the evidence for the "WithoutHelmet covers a whole
    scooter" problem: if the 95th-percentile helmet box covers a large share of the frame,
    the model (or its training labels) is drawing region-sized boxes.

It also applies the app's helmet-box filter to the same predictions and reports, per class, how many
boxes it kept and removed (and why), and saves example frames with the removed boxes drawn in grey
(--diag-dir). Look at those images: if the grey boxes are real riders the filter is hiding good
predictions; if the coloured boxes are still wrong, the filter cannot fix the model.

It does not measure accuracy: that needs labelled data (see scripts/evaluate.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from collections import Counter  # noqa: E402

from app.detector import (EXPECTED_CLASSES, HELMET_CLASSES, ClassMappingError, DetectorError,  # noqa: E402
                          ProcessingSettings, draw_diagnostic, filter_detections, load_engine,
                          validate_class_names)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/best.pt")
    ap.add_argument("--video", help="video to sample frames from")
    ap.add_argument("--frames", type=int, default=40, help="number of evenly spaced frames to sample")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/audit_report.json")
    ap.add_argument("--diag-dir", default="results/audit_diagnostics",
                    help="where to save example frames showing which boxes the filter removed")
    ap.add_argument("--diag-images", type=int, default=8)
    args = ap.parse_args()

    report: dict = {"model": args.model}
    try:
        from ultralytics import YOLO
        raw = YOLO(args.model)
    except Exception as exc:
        print(f"ERROR: cannot load model: {exc}")
        return 2
    report["class_names_in_weights"] = {int(k): v for k, v in raw.names.items()}
    print("Class names stored in the weights:", report["class_names_in_weights"])
    try:
        validate_class_names(raw.names)
        report["class_mapping_ok"] = True
        print("Class mapping: OK (matches Plate / WithHelmet / WithoutHelmet / TripleRiding)")
    except ClassMappingError as exc:
        report["class_mapping_ok"] = False
        report["class_mapping_error"] = str(exc)
        print("Class mapping: MISMATCH\n ", exc)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        return 1

    if args.video:
        engine = load_engine(args.model, args.device)
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"ERROR: cannot open video {args.video}")
            return 2
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        idxs = np.linspace(0, max(total - 1, 0), args.frames).astype(int) if total else range(args.frames)
        per_class = {c: [] for c in EXPECTED_CLASSES}   # (conf, area_fraction, aspect)
        kept_n = Counter()
        removed_n = Counter()
        reasons = Counter()
        diag_saved = 0
        diag_dir = Path(args.diag_dir)
        fsettings = ProcessingSettings()
        used = 0
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            scale = min(1.0, 1280 / w)
            if scale < 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                h, w = frame.shape[:2]
            raw = engine.infer(frame, args.conf, args.imgsz)
            kept, removed = filter_detections(raw, w, h, fsettings)
            for d in kept:
                kept_n[d.cls_id] += 1
            for d, why in removed:
                removed_n[d.cls_id] += 1
                reasons[why] += 1
            if removed and diag_saved < args.diag_images:
                diag_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(diag_dir / f"frame_{int(i):06d}.jpg"), draw_diagnostic(frame.copy(), kept, removed))
                diag_saved += 1
            for d in raw:
                x1, y1, x2, y2 = d.xyxy
                bw, bh = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
                per_class[d.cls_id].append((d.conf, bw * bh / (w * h), max(bw / bh, bh / bw)))
            used += 1
        cap.release()
        stats = {}
        print(f"\nSampled {used} frames at conf>={args.conf}, imgsz={args.imgsz}")
        print(f"{'class':14s}{'n':>6s}{'conf med':>10s}{'area med':>10s}{'area p95':>10s}{'area max':>10s}{'>8% frame':>11s}")
        for cid, name in EXPECTED_CLASSES.items():
            rows = np.array(per_class[cid]) if per_class[cid] else np.zeros((0, 3))
            n = len(rows)
            s = {"detections": n}
            if n:
                s.update(conf_median=float(np.median(rows[:, 0])), area_median=float(np.median(rows[:, 1])),
                         area_p95=float(np.percentile(rows[:, 1], 95)), area_max=float(rows[:, 1].max()),
                         frac_over_8pct=float((rows[:, 1] > 0.08).mean()))
                print(f"{name:14s}{n:6d}{s['conf_median']:10.2f}{s['area_median']:10.4f}{s['area_p95']:10.4f}"
                      f"{s['area_max']:10.4f}{s['frac_over_8pct']:11.1%}")
            else:
                print(f"{name:14s}{0:6d}")
            stats[name] = s
        report["video_sample"] = {"video": args.video, "frames_used": used, "conf": args.conf,
                                  "imgsz": args.imgsz, "per_class": stats}
        print("\nEffect of the helmet-box filter (default settings) on the same predictions")
        print(f"{'class':14s}{'raw':>6s}{'kept':>7s}{'removed':>9s}")
        effect = {}
        for cid, name in EXPECTED_CLASSES.items():
            raw_n = kept_n[cid] + removed_n[cid]
            effect[name] = {"raw": raw_n, "kept": kept_n[cid], "removed": removed_n[cid]}
            print(f"{name:14s}{raw_n:6d}{kept_n[cid]:7d}{removed_n[cid]:9d}")
        print("Reasons:", dict(reasons) if reasons else "none removed")
        if diag_saved:
            print(f"Saved {diag_saved} example frame(s) with removed boxes in grey to {diag_dir}")
        report["filter_effect"] = {"per_class": effect, "reasons": dict(reasons), "diagnostic_images": diag_saved}
        helmet_big = [stats[EXPECTED_CLASSES[c]].get("frac_over_8pct", 0) for c in HELMET_CLASSES
                      if stats[EXPECTED_CLASSES[c]]["detections"]]
        if helmet_big and max(helmet_big) > 0.1:
            print("\nNOTE: more than 10% of helmet-class boxes cover over 8% of the frame. Head-sized boxes are"
                  " normally far smaller, so these are region-sized predictions. Check the training labels with"
                  " scripts/inspect_dataset.py before deciding on retraining.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
