"""Failures that can happen while the model loads, and how they are reported.

These use stand-ins for ultralytics/torch to simulate the broken installs that really happen on Windows.
They test the application's behaviour when loading fails, not model accuracy.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import detector
from app.detector import DetectorError, load_engine
from app.main import create_app
from tests.conftest import FakeEngine
from tests.test_api import _upload

GOOD = {0: "Plate", 1: "WithHelmet", 2: "WithoutHelmet", 3: "TripleRiding"}


@pytest.fixture
def weights(tmp_path):
    p = tmp_path / "models" / "best.pt"
    p.parent.mkdir()
    p.write_bytes(b"not real weights, a stand-in file so the existence check passes")
    return p


def _fake_ultralytics(monkeypatch, *, import_error=None, yolo=None):
    mod = types.ModuleType("ultralytics")
    if import_error is not None:
        def boom(name):
            raise import_error
        mod.__getattr__ = boom
    else:
        mod.YOLO = yolo
    monkeypatch.setitem(sys.modules, "ultralytics", mod)


# ------------------------------------------------------------------ A: load failures become clear messages
def test_broken_pytorch_dll_on_windows_is_a_clear_error_not_a_crash(weights, monkeypatch):
    _fake_ultralytics(monkeypatch, import_error=OSError("[WinError 126] The specified module could not be found. Error loading c10.dll"))
    with pytest.raises(DetectorError) as e:
        load_engine(weights, "cpu")
    msg = str(e.value)
    assert "c10.dll" in msg and "PyTorch" in msg and "OSError" in msg


def test_torch_torchvision_mismatch_during_warmup_is_a_clear_error(weights, monkeypatch):
    class Model:
        names = GOOD
        def predict(self, *a, **k):
            raise RuntimeError("operator torchvision::nms does not exist")
    _fake_ultralytics(monkeypatch, yolo=lambda path: Model())
    with pytest.raises(DetectorError) as e:
        load_engine(weights, "cpu")
    assert "test inference" in str(e.value) and "torchvision::nms" in str(e.value)


def test_corrupt_weights_file_is_a_clear_error(weights, monkeypatch):
    def bad(path):
        raise RuntimeError("PytorchStreamReader failed reading zip archive")
    _fake_ultralytics(monkeypatch, yolo=bad)
    with pytest.raises(DetectorError, match="Could not load model 'best.pt'"):
        load_engine(weights, "cpu")


def test_server_still_starts_and_explains_when_the_model_install_is_broken(tmp_path, weights, monkeypatch):
    _fake_ultralytics(monkeypatch, import_error=OSError("[WinError 126] Error loading c10.dll"))
    app = create_app(model_path=weights, upload_dir=tmp_path / "up", results_dir=tmp_path / "res")
    with TestClient(app) as c:                         # must not raise during startup
        h = c.get("/api/health").json()
        assert h["model_loaded"] is False and "c10.dll" in h["model_error"]
        assert h["model_path_exists"] is True           # the file is fine; the install is not
        vid = _upload(c, _short_video(tmp_path)).json()["video_id"]
        r = c.post("/api/jobs", json={"video_id": vid})
        assert r.status_code == 503 and "c10.dll" in r.json()["detail"]


def test_unexpected_exception_type_while_loading_does_not_crash_startup(tmp_path, weights, monkeypatch):
    import app.main as m
    monkeypatch.setattr(m, "load_engine", lambda *a, **k: (_ for _ in ()).throw(MemoryError("out of memory")))
    app = create_app(model_path=weights, upload_dir=tmp_path / "up", results_dir=tmp_path / "res")
    with TestClient(app) as c:
        h = c.get("/api/health").json()
        assert h["model_loaded"] is False and "MemoryError" in h["model_error"]


def _short_video(tmp_path):
    from tests.conftest import make_video
    return make_video(tmp_path / "v.mp4")


# ------------------------------------------------------------------ B: health notices a model that appears later
def test_health_notices_weights_placed_while_server_is_running(tmp_path, monkeypatch):
    import app.main as m
    calls = []
    def fake_load(path, dev):                      # like the real loader: refuses a missing file
        calls.append(path)
        if not Path(path).is_file():
            raise detector.ModelNotFoundError(f"Model file not found: {path}. Set HELMET_MODEL_PATH.")
        return FakeEngine()
    monkeypatch.setattr(m, "load_engine", fake_load)
    weights = tmp_path / "models" / "best.pt"
    app = create_app(model_path=weights, upload_dir=tmp_path / "up", results_dir=tmp_path / "res")
    with TestClient(app) as c:
        assert len(calls) == 1                                           # the startup attempt
        for _ in range(3):
            h = c.get("/api/health").json()
            assert h["model_loaded"] is False and h["model_path_exists"] is False
        assert len(calls) == 1                                           # a missing file is not retried
        weights.parent.mkdir(); weights.write_bytes(b"x" * 100)       # the user copies best.pt in
        h = c.get("/api/health").json()
        assert h["model_loaded"] is True and h["model_error"] is None and len(calls) == 2
        assert c.get("/api/health").json()["model_loaded"] and len(calls) == 2      # not reloaded again


def test_health_does_not_retry_an_unchanged_broken_file_on_every_request(tmp_path, monkeypatch):
    import app.main as m
    calls = []
    def failing(path, dev):
        calls.append(1)
        raise DetectorError("Could not load model 'best.pt': corrupt")
    monkeypatch.setattr(m, "load_engine", failing)
    weights = tmp_path / "best.pt"; weights.write_bytes(b"broken")
    app = create_app(model_path=weights, upload_dir=tmp_path / "up", results_dir=tmp_path / "res")
    with TestClient(app) as c:
        for _ in range(4):
            assert c.get("/api/health").json()["model_error"].endswith("corrupt")
        assert len(calls) == 1                                          # startup attempt only
        weights.write_bytes(b"replaced with a different file")
        c.get("/api/health")
        assert len(calls) == 2                                          # file changed, so it tries once more


# ------------------------------------------------------------------ C: health says where it is looking
def test_health_reports_path_existence_size_device_and_encoder(tmp_path, fake_engine):
    weights = tmp_path / "models" / "best.pt"
    weights.parent.mkdir(); weights.write_bytes(b"x" * 2048)
    with TestClient(create_app(engine=fake_engine, model_path=weights, upload_dir=tmp_path / "u", results_dir=tmp_path / "r")) as c:
        h = c.get("/api/health").json()
    assert h["model_path"] == str(weights) and h["model_path_exists"] is True
    assert h["model_size_mb"] == pytest.approx(2048 / 1e6, abs=0.001)
    assert h["device"] == "cpu" and h["device_requested"] == "cpu"
    assert isinstance(h["h264_encoder_available"], bool)
    assert h["model_loaded"] is True and h["model_error"] is None


def test_health_for_missing_weights_says_exactly_where_to_put_them(tmp_path):
    weights = tmp_path / "models" / "best.pt"
    with TestClient(create_app(model_path=weights, upload_dir=tmp_path / "u", results_dir=tmp_path / "r")) as c:
        h = c.get("/api/health").json()
    assert h["model_loaded"] is False and h["model_path_exists"] is False
    assert str(weights) in h["model_error"] and "HELMET_MODEL_PATH" in h["model_error"]
    assert h["model_size_mb"] is None and h["device"] is None
