"""Central configuration. Every value can be overridden with an environment variable."""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = Path(os.getenv("HELMET_MODEL_PATH", BASE_DIR / "models" / "best.pt"))
UPLOAD_DIR = Path(os.getenv("HELMET_UPLOAD_DIR", BASE_DIR / "uploads"))
RESULTS_DIR = Path(os.getenv("HELMET_RESULTS_DIR", BASE_DIR / "results"))

# "cpu" (default), "cuda" (used only if a CUDA GPU is actually available) or "auto".
DEVICE = os.getenv("HELMET_DEVICE", "cpu").lower()

# Delete uploads and results created by this app once older than this many hours. 0 = never.
RETENTION_HOURS = float(os.getenv("HELMET_RETENTION_HOURS", "0"))
MAX_UPLOAD_MB = int(os.getenv("HELMET_MAX_UPLOAD_MB", "500"))
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# Class mapping the model was trained with (Kaggle dataset: traffic violations,
# triple riding, no helmet, plate). The loader refuses a model that disagrees.
EXPECTED_CLASSES = {
    0: "Plate",
    1: "WithHelmet",
    2: "WithoutHelmet",
    3: "TripleRiding",
}
