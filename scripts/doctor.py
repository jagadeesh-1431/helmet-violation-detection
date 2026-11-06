"""Check everything the app needs and say exactly what is wrong. Changes nothing.

    python scripts/doctor.py                                  # install, model, video writing
    python scripts/doctor.py --video videos/sample_clip.mp4   # also run the model on one real frame

Each line is PASS, WARN or FAIL. A FAIL comes with the fix. The exit code is 1 if anything FAILED.

What this can and cannot tell you: it proves the model file loads, has the right classes, and runs on a real
frame, and that this PC can write and encode video. It cannot tell you the model is ACCURATE: zero boxes on
one frame can simply mean there was nothing the model recognises in that frame.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    fix: str = ""


REQUIRED = [("ultralytics", "ultralytics"), ("torch", "torch"), ("torchvision", "torchvision"),
            ("cv2", "opencv-python"), ("numpy", "numpy"), ("fastapi", "fastapi"), ("uvicorn", "uvicorn"),
            ("multipart", "python-multipart"), ("jinja2", "jinja2"), ("yaml", "PyYAML"),
            ("imageio_ffmpeg", "imageio-ffmpeg")]
OPTIONAL = [("lap", "lapx", "only needed for 'Estimate distinct tracks'")]


def check_python() -> Check:
    v = sys.version_info
    ver = f"{v.major}.{v.minor}.{v.micro}"
    if v < (3, 9):
        return Check("Python", FAIL, ver, "Use Python 3.9 to 3.12 and recreate .venv with that version.")
    return Check("Python", PASS, ver)


def check_packages() -> list[Check]:
    out = []
    for mod, pip_name in REQUIRED:
        try:
            m = importlib.import_module(mod)
            out.append(Check(f"package {pip_name}", PASS, str(getattr(m, "__version__", "installed"))))
        except ModuleNotFoundError:
            out.append(Check(f"package {pip_name}", FAIL, "not installed", "pip install -r requirements.txt"))
        except Exception as exc:   # e.g. OSError: [WinError 126] Error loading c10.dll
            out.append(Check(f"package {pip_name}", FAIL, f"installed but cannot start: {type(exc).__name__}: {exc}",
                             "Reinstall PyTorch for this Python version, then: pip install -r requirements.txt"))
    for mod, pip_name, why in OPTIONAL:
        try:
            importlib.import_module(mod)
            out.append(Check(f"package {pip_name}", PASS, "installed (tracking available)"))
        except Exception:
            out.append(Check(f"package {pip_name}", WARN, f"not installed ({why})", f"pip install {pip_name}"))
    return out


def resolve_model_path(arg: str | None) -> tuple[Path, str]:
    if arg:
        return Path(arg), "--model argument"
    env = os.getenv("HELMET_MODEL_PATH")
    if env:
        return Path(env), "HELMET_MODEL_PATH environment variable"
    from app import config
    return Path(config.MODEL_PATH), "default (models/best.pt)"


def check_model_file(path: Path, source: str) -> Check:
    if not path.is_file():
        return Check("model file", FAIL, f"not found: {path.resolve()}  (from {source})",
                     f"Copy your trained best.pt to {path.resolve()} (or set HELMET_MODEL_PATH). "
                     "The app never substitutes another model.")
    size = path.stat().st_size
    if not os.access(path, os.R_OK):
        return Check("model file", FAIL, f"exists but cannot be read: {path}", "Check the file permissions.")
    note = f"{path.resolve()}  ({size / 1e6:.1f} MB, from {source})"
    if size < 1_000_000:
        return Check("model file", WARN, note + " - unusually small for a YOLO model", "Check this is the real best.pt, not a partial copy.")
    return Check("model file", PASS, note)


def check_model_loads(path: Path, device: str):
    """Returns (Check, engine or None)."""
    from app.detector import ClassMappingError, DetectorError, load_engine
    try:
        engine = load_engine(path, device)
    except ClassMappingError as exc:
        return Check("model loads and classes match", FAIL, str(exc),
                     "This is not the 4-class helmet model (for example yolov8n.pt). Use the best.pt from your training run."), None
    except DetectorError as exc:
        return Check("model loads and classes match", FAIL, str(exc), "See the message above for the cause."), None
    names = {int(k): v for k, v in engine.model.names.items()}
    return Check("model loads and classes match", PASS, f"device {engine.device}; classes {names}"), engine


def check_video_io() -> list[Check]:
    """Can this PC write an MP4 and encode it for browsers? Uses a tiny generated clip in a temp folder."""
    import cv2
    import numpy as np
    from app.detector import _transcode_h264, find_ffmpeg
    out = []
    with tempfile.TemporaryDirectory() as td:
        raw, enc = Path(td) / "raw.mp4", Path(td) / "enc.mp4"
        w = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
        if not w.isOpened():
            return [Check("write MP4 with OpenCV", FAIL, "VideoWriter could not open", "Reinstall opencv-python (not opencv-python-headless).")]
        for i in range(10):
            w.write(np.full((64, 64, 3), i * 20, np.uint8))
        w.release()
        cap = cv2.VideoCapture(str(raw))
        n = 0
        while cap.grab():
            n += 1
        cap.release()
        out.append(Check("write MP4 with OpenCV", PASS if n == 10 else FAIL, f"wrote 10 frames, read back {n}"))
        if not find_ffmpeg():
            out.append(Check("H.264 encoder (browser playback)", WARN, "no ffmpeg found",
                             "pip install imageio-ffmpeg  (without it the result still downloads but browsers may not play it)"))
        elif _transcode_h264(raw, enc):
            out.append(Check("H.264 encoder (browser playback)", PASS, f"ffmpeg at {find_ffmpeg()}"))
        else:
            out.append(Check("H.264 encoder (browser playback)", WARN, "ffmpeg found but the test encode failed",
                             "Reinstall imageio-ffmpeg; the result will still download but may not play in the browser."))
    return out


def check_folders() -> list[Check]:
    from app import config
    out = []
    for label, d in (("uploads folder", config.UPLOAD_DIR), ("results folder", config.RESULTS_DIR)):
        try:
            Path(d).mkdir(parents=True, exist_ok=True)
            probe = Path(d) / ".write_test"
            probe.write_text("x")
            probe.unlink()
            out.append(Check(label, PASS, f"writable: {Path(d).resolve()}"))
        except OSError as exc:
            out.append(Check(label, FAIL, f"cannot write to {d}: {exc}", "Check the folder permissions or set HELMET_UPLOAD_DIR / HELMET_RESULTS_DIR."))
    return out


def check_one_frame(engine, video: Path, conf: float, imgsz: int) -> list[Check]:
    import cv2
    from app.detector import EmptyVideoError, InvalidVideoError, ProcessingSettings, VideoNotFoundError, filter_detections, probe_video
    try:
        info = probe_video(video)
    except (VideoNotFoundError, InvalidVideoError, EmptyVideoError) as exc:
        return [Check("video readable", FAIL, str(exc), "Try another video, or convert it to MP4 (H.264).")]
    out = [Check("video readable", PASS, f"{info['width']}x{info['height']} at {info['fps']} FPS, {info['frame_count']} frames")]
    cap = cv2.VideoCapture(str(video))
    if info["frame_count"]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, info["frame_count"] // 2)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return out + [Check("read a frame", FAIL, "could not decode a frame from the middle of the video", "Try another video.")]
    if frame.shape[1] > 1280:
        s = 1280 / frame.shape[1]
        frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    h, w = frame.shape[:2]
    try:
        raw = engine.infer(frame, conf, imgsz)
    except Exception as exc:
        return out + [Check("run the model on one frame", FAIL, f"{type(exc).__name__}: {exc}", "Send this message to whoever is helping you debug.")]
    kept, removed = filter_detections(raw, w, h, ProcessingSettings())
    if raw:
        desc = "; ".join(f"{d.cls_name} {d.conf:.2f} (box {(d.xyxy[2]-d.xyxy[0])*(d.xyxy[3]-d.xyxy[1])/(w*h):.1%} of frame)" for d in raw[:8])
        out.append(Check("run the model on one frame", PASS, f"{len(raw)} box(es) at confidence {conf}: {desc}"))
        out.append(Check("helmet filter on this frame", PASS, f"kept {len(kept)}, removed {len(removed)}" +
                         (f" ({', '.join(sorted({r for _, r in removed}))})" if removed else "")))
        return out
    probes = []
    for lower in (0.10, 0.05, 0.01):
        if lower < conf:
            n = len(engine.infer(frame, lower, imgsz))
            probes.append(f"{n} at {lower}")
    out.append(Check("run the model on one frame", WARN,
                     f"no boxes at confidence {conf} (the model ran without error). Lower thresholds: {', '.join(probes) or 'n/a'}",
                     "Zero boxes on ONE frame is not a failure by itself. Try a frame with a visible rider, or run "
                     "scripts/audit_model.py over 40 frames. If boxes only appear at very low confidence, the model is unsure on this footage."))
    return out


def run_checks(args) -> list[Check]:
    checks = [check_python()]
    checks += check_packages()
    checks += check_folders()
    checks += check_video_io() if not any(c.status == FAIL and c.name in ("package opencv-python", "package numpy") for c in checks) else []
    path, source = resolve_model_path(args.model)
    mf = check_model_file(path, source)
    checks.append(mf)
    engine = None
    blocked = any(c.status == FAIL and c.name in ("package ultralytics", "package torch", "package torchvision") for c in checks)
    if mf.status == FAIL:
        checks.append(Check("model loads and classes match", SKIP, "skipped: the model file is missing"))
    elif blocked:
        checks.append(Check("model loads and classes match", SKIP, "skipped: PyTorch/Ultralytics is not working (see above)"))
    else:
        c, engine = check_model_loads(path, args.device)
        checks.append(c)
    if args.video:
        if engine is None:
            checks.append(Check("run the model on one frame", SKIP, "skipped: the model did not load"))
        else:
            checks += check_one_frame(engine, Path(args.video), args.conf, args.imgsz)
    else:
        checks.append(Check("run the model on one frame", SKIP, "pass --video <file> to test real inference"))
    return checks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="weights file (default: HELMET_MODEL_PATH or models/best.pt)")
    ap.add_argument("--video", help="a video to take one frame from for a real inference test")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    checks = run_checks(args)
    for c in checks:
        print(f"[{c.status}] {c.name}: {c.detail}")
        if c.fix and c.status in (FAIL, WARN):
            print(f"       -> {c.fix}")
    fails = [c for c in checks if c.status == FAIL]
    warns = [c for c in checks if c.status == WARN]
    print()
    if fails:
        print(f"{len(fails)} problem(s) must be fixed before detection can work. Start with the first FAIL above.")
    else:
        print("No blocking problems found." + (f" {len(warns)} warning(s) above." if warns else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
