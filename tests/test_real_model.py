"""Tests that load real YOLO weights. They are skipped (not faked) when the weights or the
ultralytics package are missing.

* models/best.pt  -> your trained model: loading, class mapping, end-to-end processing.
* models/yolov8n.pt (COCO) -> PLUMBING ONLY: proves our predict/track calls work with the real
  ultralytics API. Its class names are not ours, so its detections mean nothing for this project.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import pytest

pytest.importorskip("ultralytics")

from app.detector import (ClassMappingError, EXPECTED_CLASSES, ProcessingSettings, YoloEngine,  # noqa: E402
                          load_engine, process_video)

ROOT = Path(__file__).resolve().parent.parent
BEST = ROOT / "models" / "best.pt"
STOCK = ROOT / "models" / "yolov8n.pt"

needs_best = pytest.mark.skipif(not BEST.is_file(), reason="models/best.pt not present")
needs_stock = pytest.mark.skipif(not STOCK.is_file(), reason="models/yolov8n.pt not present")


@needs_best
def test_trained_model_loads_with_correct_class_mapping():
    engine = load_engine(BEST, "cpu")
    assert engine.device == "cpu"
    assert {int(k): v for k, v in engine.model.names.items()} == EXPECTED_CLASSES


@needs_best
def test_trained_model_processes_short_video(tmp_path, short_video):
    engine = load_engine(BEST, "cpu")
    res = process_video(engine, short_video, tmp_path / "o.mp4", ProcessingSettings(conf=0.25))
    assert res["output"]["frames"] == 20
    assert cv2.VideoCapture(str(tmp_path / "o.mp4")).isOpened()


@needs_stock
def test_stock_model_is_rejected_by_class_mapping_check():
    with pytest.raises(ClassMappingError):
        load_engine(STOCK, "cpu")


@needs_stock
@pytest.mark.parametrize("track", [False, True])
def test_engine_plumbing_with_stock_model(tmp_path, short_video, track):
    from ultralytics import YOLO
    engine = YoloEngine(YOLO(str(STOCK)), "cpu", STOCK)      # bypasses the mapping check on purpose
    res = process_video(engine, short_video, tmp_path / "o.mp4", ProcessingSettings(track=track, conf=0.05))
    assert res["frames_read"] == 20 and res["output"]["frames"] == 20
    assert res["counts"]["frames_inferred"] == 20


@needs_best
def test_audit_script_runs_on_trained_model(tmp_path, short_video):
    import json, subprocess, sys
    out = tmp_path / "audit.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "audit_model.py"), "--model", str(BEST),
                        "--video", str(short_video), "--frames", "5", "--out", str(out),
                        "--diag-dir", str(tmp_path / "d")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.loads(out.read_text())
    assert rep["class_mapping_ok"] is True and "filter_effect" in rep
