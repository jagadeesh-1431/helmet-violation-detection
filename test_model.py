"""Verify that the bundled trained YOLO model loads and has the expected classes."""
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = Path(__file__).resolve().parent / "models" / "best.pt"
EXPECTED = {0: "Plate", 1: "WithHelmet", 2: "WithoutHelmet", 3: "TripleRiding"}

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))
actual = {int(k): str(v) for k, v in model.names.items()}
print("Model loaded:", MODEL_PATH)
print("Classes:", actual)
if actual != EXPECTED:
    raise SystemExit(f"Unexpected class mapping. Expected {EXPECTED}, got {actual}")
print("Model check passed.")
