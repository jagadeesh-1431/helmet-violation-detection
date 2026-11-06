"""Shared fixtures.

NOTE: most tests use FakeEngine, a scripted stand-in that returns fixed detections. Those tests
verify the pipeline, API and counting logic. They say nothing about the accuracy of the trained
model. Tests against the real weights live in test_real_model.py and are skipped when
models/best.pt is not present.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.detector import Detection  # noqa: E402


def make_video(path: Path, frames: int = 20, width: int = 320, height: int = 240, fps: float = 10.0) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened(), "test environment cannot write mp4v videos"
    for i in range(frames):
        img = np.full((height, width, 3), 60, dtype=np.uint8)
        x = 10 + i * 5
        cv2.rectangle(img, (x, 80), (x + 40, 140), (200, 200, 200), -1)
        writer.write(img)
    writer.release()
    return path


class FakeEngine:
    """Returns scripted detections. Records how it was called."""
    device = "cpu"

    def __init__(self):
        self.calls = 0
        self.reset_calls = 0

    def reset_tracking(self):
        self.reset_calls += 1

    def infer(self, frame, conf, imgsz, track=False):
        self.calls += 1
        n = self.calls
        dets = [
            Detection(0, "Plate", 0.9, (10, 10, 40, 25), 1 if track else None),
            Detection(2, "WithoutHelmet", 0.5 + (n % 5) / 20, (100, 40, 130, 70), 2 if track else None),
        ]
        if n <= 5:
            dets.append(Detection(3, "TripleRiding", 0.8, (60, 30, 200, 200), 3 if track else None))
        if n == 7:  # a whole-frame "WithoutHelmet" box: must be suppressed by the sanity filter
            dets.append(Detection(2, "WithoutHelmet", 0.95, (0, 0, 300, 230)))
        return dets


@pytest.fixture
def short_video(tmp_path) -> Path:
    return make_video(tmp_path / "short.mp4")


@pytest.fixture
def fake_engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def client(tmp_path, fake_engine):
    from fastapi.testclient import TestClient
    from app.main import create_app
    app = create_app(engine=fake_engine, upload_dir=tmp_path / "up", results_dir=tmp_path / "res",
                     model_path=tmp_path / "unused.pt")
    with TestClient(app) as c:
        yield c
