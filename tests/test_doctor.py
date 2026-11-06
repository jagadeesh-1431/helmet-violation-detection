from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import doctor  # noqa: E402

from app.detector import Detection  # noqa: E402
from app.main import create_app  # noqa: E402
from tests.conftest import FakeEngine, make_video  # noqa: E402
from tests.test_api import _upload, _wait  # noqa: E402

GOOD = {0: "Plate", 1: "WithHelmet", 2: "WithoutHelmet", 3: "TripleRiding"}


def _status(checks, name):
    return next(c for c in checks if c.name == name)


def _args(**kw):
    base = dict(model=None, video=None, conf=0.25, imgsz=640, device="cpu")
    base.update(kw)
    return types.SimpleNamespace(**base)


# ------------------------------------------------------------------ model file
def test_missing_model_says_exactly_where_to_put_it(tmp_path):
    c = doctor.check_model_file(tmp_path / "models" / "best.pt", "default")
    assert c.status == doctor.FAIL and str(tmp_path / "models" / "best.pt") in c.detail
    assert "Copy your trained best.pt" in c.fix and "never substitutes" in c.fix


def test_tiny_model_file_gets_a_warning_not_a_pass(tmp_path):
    p = tmp_path / "best.pt"; p.write_bytes(b"x" * 100)
    assert doctor.check_model_file(p, "default").status == doctor.WARN


def test_model_path_source_is_reported(monkeypatch, tmp_path):
    monkeypatch.setenv("HELMET_MODEL_PATH", str(tmp_path / "other.pt"))
    path, source = doctor.resolve_model_path(None)
    assert path == tmp_path / "other.pt" and "HELMET_MODEL_PATH" in source
    assert doctor.resolve_model_path("x.pt") == (Path("x.pt"), "--model argument")


# ------------------------------------------------------------------ loading problems
def _fake_ultralytics(monkeypatch, **attrs):
    mod = types.ModuleType("ultralytics")
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, "ultralytics", mod)


def test_wrong_model_is_rejected_with_an_explanation(tmp_path, monkeypatch):
    p = tmp_path / "best.pt"; p.write_bytes(b"x" * 10)
    coco = types.SimpleNamespace(names={i: f"coco{i}" for i in range(80)}, predict=lambda *a, **k: [])
    _fake_ultralytics(monkeypatch, YOLO=lambda path: coco)
    c, engine = doctor.check_model_loads(p, "cpu")
    assert engine is None and c.status == doctor.FAIL
    assert "class mapping" in c.detail.lower() and "yolov8n.pt" in c.fix


def test_broken_pytorch_install_is_reported_not_raised(tmp_path, monkeypatch):
    p = tmp_path / "best.pt"; p.write_bytes(b"x" * 10)
    mod = types.ModuleType("ultralytics")
    def boom(name):
        raise OSError("[WinError 126] Error loading c10.dll")
    mod.__getattr__ = boom
    monkeypatch.setitem(sys.modules, "ultralytics", mod)
    c, engine = doctor.check_model_loads(p, "cpu")
    assert engine is None and c.status == doctor.FAIL and "c10.dll" in c.detail


def test_full_run_with_missing_model_fails_and_skips_dependent_checks(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("HELMET_MODEL_PATH", raising=False)
    rc = doctor.main(["--model", str(tmp_path / "nope.pt"), "--video", str(make_video(tmp_path / "v.mp4"))])
    out = capsys.readouterr().out
    assert rc == 1
    assert "[FAIL] model file" in out and "[SKIP] model loads and classes match" in out
    assert "[SKIP] run the model on one frame" in out and "Start with the first FAIL" in out


# ------------------------------------------------------------------ video writing
def test_video_io_check_passes_here_and_reports_the_encoder():
    checks = doctor.check_video_io()
    assert _status(checks, "write MP4 with OpenCV").status == doctor.PASS
    assert _status(checks, "H.264 encoder (browser playback)").status in (doctor.PASS, doctor.WARN)


# ------------------------------------------------------------------ one real frame
def test_one_frame_reports_actual_boxes_and_filter_effect(tmp_path):
    video = make_video(tmp_path / "v.mp4")
    checks = doctor.check_one_frame(FakeEngine(), video, 0.25, 640)
    assert _status(checks, "video readable").status == doctor.PASS
    run = _status(checks, "run the model on one frame")
    assert run.status == doctor.PASS and "Plate 0.90" in run.detail and "WithoutHelmet" in run.detail
    assert "kept 3, removed 0" in _status(checks, "helmet filter on this frame").detail   # Plate, WithoutHelmet, TripleRiding


def test_zero_boxes_is_a_warning_with_threshold_evidence_not_a_fake_pass(tmp_path):
    class Quiet:
        def infer(self, frame, conf, imgsz, track=False):
            return [Detection(2, "WithoutHelmet", 0.04, (10, 10, 40, 40))] if conf <= 0.05 else []
    checks = doctor.check_one_frame(Quiet(), make_video(tmp_path / "v.mp4"), 0.25, 640)
    run = _status(checks, "run the model on one frame")
    assert run.status == doctor.WARN and "no boxes at confidence 0.25" in run.detail
    assert "0 at 0.1" in run.detail and "1 at 0.05" in run.detail and "1 at 0.01" in run.detail
    assert "not a failure by itself" in run.fix


def test_unreadable_video_is_a_clear_failure(tmp_path):
    bad = tmp_path / "bad.mp4"; bad.write_text("not a video")
    c = doctor.check_one_frame(FakeEngine(), bad, 0.25, 640)[0]
    assert c.status == doctor.FAIL and c.name == "video readable"


# ------------------------------------------------------------------ zero detections through the whole app
def test_video_with_no_detections_completes_honestly(tmp_path, short_video):
    class Silent:
        device = "cpu"
        def infer(self, *a, **k):
            return []
    app = create_app(engine=Silent(), upload_dir=tmp_path / "up", results_dir=tmp_path / "res")
    with TestClient(app) as c:
        vid = _upload(c, short_video).json()["video_id"]
        job = _wait(c, c.post("/api/jobs", json={"video_id": vid, "diagnostics": True}).json()["job_id"])
        assert job["status"] == "completed"
        res = job["result"]
        assert all(row["detections"] == 0 and row["mean_confidence"] is None for row in res["counts"]["classes"])
        assert res["flagged_for_review"] == [] and res["output"]["frames"] == 20
        assert c.get(job["video_url"]).status_code == 200
        assert res["diagnostics"]["frames_with_removed_boxes"] == 0 and res["diagnostics"]["samples"] == []


def test_results_page_explains_zero_detections_without_inventing_any():
    js = (ROOT / "app" / "static" / "results.js").read_text(encoding="utf-8")
    assert "drew no boxes" in js and "does not mean the video has no riders" in js
    assert "total === 0" in js and "$(\"zero-note\")" in js
