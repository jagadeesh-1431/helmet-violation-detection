from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate as ev  # noqa: E402

# Rows = predicted (Plate, WithHelmet, WithoutHelmet, TripleRiding, background),
# columns = actual (same order). Worked out by hand below.
MATRIX = [
    [8, 0, 0, 0, 2],     # predicted Plate: 8 right, 2 false alarms on background
    [0, 5, 1, 0, 1],     # predicted WithHelmet: 5 right, 1 was WithoutHelmet, 1 false alarm
    [0, 2, 3, 0, 0],     # predicted WithoutHelmet: 3 right, 2 were WithHelmet
    [0, 0, 0, 4, 0],     # predicted TripleRiding: all 4 right
    [1, 1, 2, 1, 0],     # background row: objects the model missed
]


def test_operating_point_matches_hand_calculation():
    op = ev.operating_point(MATRIX)
    assert op["Plate"] == {"tp": 8, "fp": 2, "fn": 1, "precision": 0.8, "recall": 0.8889}
    assert op["WithHelmet"] == {"tp": 5, "fp": 2, "fn": 3, "precision": 0.7143, "recall": 0.625}
    assert op["WithoutHelmet"] == {"tp": 3, "fp": 2, "fn": 3, "precision": 0.6, "recall": 0.5}
    assert op["TripleRiding"] == {"tp": 4, "fp": 0, "fn": 1, "precision": 1.0, "recall": 0.8}


def test_operating_point_handles_empty_classes_without_dividing_by_zero():
    op = ev.operating_point([[0] * 5 for _ in range(5)])
    assert all(v["precision"] is None and v["recall"] is None for v in op.values())


def _report(op=True):
    return {
        "generated_at": "2026-01-01 00:00",
        "model": {"file": "best.pt", "size_mb": 6.2, "sha256_12": "abc123abc123"},
        "dataset": {"data_yaml": "data/raw/data.yaml", "split": "val", "images": 100},
        "settings": {"imgsz": 640, "device": "cpu", "map_conf": 0.001, "map_iou": 0.6, "cm_conf": 0.25, "cm_iou": 0.45},
        "overall": {"precision": 0.7, "recall": 0.6, "map50": 0.65, "map50_95": 0.4},
        "per_class": {n: {"instances": 10, "precision": 0.7, "recall": 0.6, "map50": 0.65, "map50_95": 0.4} for n in ev.NAMES},
        "operating_point": {"conf": 0.25, "per_class": ev.operating_point(MATRIX)} if op else None,
        "confusion_matrix": {"orientation": "x", "labels": ev.NAMES + ["background"], "matrix": MATRIX} if op else None,
        "speed_ms_per_image": {"preprocess": 1.0, "inference": 50.0, "postprocess": 2.0},
        "hardware": {"processor": "cpu", "cpu_count": 8, "platform": "Windows", "python": "3.12", "torch": "2.x", "ultralytics": "8.x"},
    }


def test_markdown_contains_measured_values_and_cautions():
    md = ev.build_markdown(_report())
    assert "| WithHelmet | 5 | 2 | 3 | 0.714 | 0.625 |" in md
    assert "| **WithoutHelmet** | 0 | 2 | 3 | 0 | 0 |" in md
    assert "Rows are what the model **predicted**" in md
    assert "optimistic" in md                                   # split is 'val'
    assert "not generated here" in md and "sha256 `abc123abc123...`" in md


def test_markdown_has_no_optimism_warning_for_test_split():
    rep = _report(); rep["dataset"]["split"] = "test"
    assert "optimistic" not in ev.build_markdown(rep)


def test_markdown_omits_counts_when_confusion_matrix_unavailable():
    md = ev.build_markdown(_report(op=False))
    assert "Not available" in md and "TP | FP | FN" not in md and "predicted \\ actual" not in md
    assert "Speed and hardware" in md                           # the rest of the report is still produced


def test_cli_reports_missing_inputs_clearly(tmp_path):
    pytest.importorskip("ultralytics")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--data", str(tmp_path / "nope.yaml"),
                        "--model", str(tmp_path / "nope.pt")], capture_output=True, text=True)
    assert r.returncode == 2 and "model not found" in r.stdout
    (tmp_path / "m.pt").write_bytes(b"x")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--data", str(tmp_path / "nope.yaml"),
                        "--model", str(tmp_path / "m.pt")], capture_output=True, text=True)
    assert r.returncode == 2 and "data.yaml not found" in r.stdout
