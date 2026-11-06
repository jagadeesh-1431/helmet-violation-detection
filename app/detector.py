"""YOLO inference and video processing for helmet / triple-riding detection.

Design notes
------------
* Everything the UI reports comes from real model output. Nothing is hard-coded.
* Counts are *frame-level detections*. A rider visible in 200 frames contributes up to
  200 detections. They are NOT unique riders or unique violations.
* Optional ByteTrack tracking adds a "distinct track IDs" figure. It is an estimate:
  IDs can switch or fragment, so it is not a verified violation count either.
* Plate *detection* (a box) is not plate *reading* (OCR). OCR is not implemented.
"""
from __future__ import annotations

import json
import logging
import math
import re
import shutil
import subprocess
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from . import config

log = logging.getLogger(__name__)

EXPECTED_CLASSES = dict(config.EXPECTED_CLASSES)
HELMET_CLASSES = {1, 2}      # WithHelmet, WithoutHelmet: expected to be small head boxes
FLAG_CLASSES = {2, 3}        # WithoutHelmet, TripleRiding: queued for human review
CLASS_COLORS_BGR = {
    0: (237, 111, 47),    # Plate         blue
    1: (90, 158, 30),     # WithHelmet    green
    2: (54, 54, 217),     # WithoutHelmet red
    3: (0, 138, 232),     # TripleRiding  orange
}


# --------------------------------------------------------------------------- errors
class DetectorError(Exception):
    """Base class. The message is safe to show to the user."""


class ModelNotFoundError(DetectorError):
    pass


class ClassMappingError(DetectorError):
    pass


class VideoNotFoundError(DetectorError):
    pass


class InvalidVideoError(DetectorError):
    pass


class EmptyVideoError(DetectorError):
    pass


class VideoWriterError(DetectorError):
    pass


class ProcessingCancelled(DetectorError):
    pass


# --------------------------------------------------------------------------- data
@dataclass
class Detection:
    cls_id: int
    cls_name: str
    conf: float
    xyxy: tuple[float, float, float, float]
    track_id: Optional[int] = None


@dataclass
class ProcessingSettings:
    conf: float = 0.25
    imgsz: int = 640
    frame_stride: int = 1            # run the model on every Nth frame
    track: bool = False              # ByteTrack, adds approximate distinct-track counts
    sanity_filter: bool = True       # drop implausible helmet-class boxes (see filter_detections)
    max_output_width: int = 1280     # frames are downscaled to this width (never upscaled); 0 = keep original size
    diagnostics: bool = False        # save raw vs filtered predictions and the boxes the filter removed
    diagnostic_samples_max: int = 12
    max_helmet_box_area_frac: float = 0.08
    max_helmet_aspect: float = 2.5
    flagged_frames_max: int = 12
    flagged_min_gap_seconds: float = 2.0

    def validate(self) -> None:
        if not 0.01 <= self.conf <= 0.99:
            raise ValueError("confidence must be between 0.01 and 0.99")
        if not 160 <= self.imgsz <= 1920:
            raise ValueError("imgsz must be between 160 and 1920")
        if not 1 <= self.frame_stride <= 60:
            raise ValueError("frame_stride must be between 1 and 60")
        if self.max_output_width != 0 and self.max_output_width < 320:
            raise ValueError("max_output_width must be 0 (original size) or at least 320")


# --------------------------------------------------------------------------- class names
def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def validate_class_names(names) -> None:
    """Raise ClassMappingError unless ``names`` matches the expected 4-class mapping."""
    if isinstance(names, (list, tuple)):
        names = dict(enumerate(names))
    names = {int(k): v for k, v in dict(names).items()}
    problems = []
    if len(names) != len(EXPECTED_CLASSES):
        problems.append(f"model has {len(names)} classes, expected {len(EXPECTED_CLASSES)}")
    for cid, expected in EXPECTED_CLASSES.items():
        got = names.get(cid)
        if got is None or _norm(got) != _norm(expected):
            problems.append(f"class {cid}: expected '{expected}', model has '{got}'")
    if problems:
        raise ClassMappingError(
            "Model class mapping does not match this project: " + "; ".join(problems)
            + ". Check that the weights were trained on the 4-class dataset."
        )


def resolve_device(requested: str) -> str:
    """CPU by default. A GPU is used only when asked for *and* actually available."""
    requested = (requested or "cpu").lower()
    if requested == "cpu":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:  # pragma: no cover - torch missing or broken
        pass
    if requested == "cuda":
        log.warning("CUDA requested but no GPU is available; falling back to CPU.")
    return "cpu"


# --------------------------------------------------------------------------- engine
def extract_detections(result) -> list[Detection]:
    """Convert one Ultralytics result into Detection objects."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    ids = boxes.id.cpu().numpy().astype(int) if getattr(boxes, "id", None) is not None else None
    out = []
    for i in range(len(cls)):
        cid = int(cls[i])
        if cid not in EXPECTED_CLASSES:
            continue
        out.append(Detection(
            cls_id=cid, cls_name=EXPECTED_CLASSES[cid], conf=float(conf[i]),
            xyxy=tuple(float(v) for v in xyxy[i]),
            track_id=int(ids[i]) if ids is not None else None,
        ))
    return out


class YoloEngine:
    """Thin wrapper around an Ultralytics model. Single-threaded use (guarded by a lock)."""

    def __init__(self, model, device: str, model_path: Path):
        self.model = model
        self.device = device
        self.model_path = Path(model_path)
        self._lock = threading.Lock()

    @property
    def class_names(self) -> dict[int, str]:
        return dict(EXPECTED_CLASSES)

    def reset_tracking(self) -> None:
        with self._lock:
            if getattr(self.model, "predictor", None) is not None:
                self.model.predictor = None  # forces a fresh tracker on the next call

    def infer(self, frame: np.ndarray, conf: float, imgsz: int, track: bool = False) -> list[Detection]:
        with self._lock:
            kwargs = dict(conf=conf, imgsz=imgsz, device=self.device, verbose=False)
            if track:
                try:
                    results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", **kwargs)
                except ImportError as exc:
                    raise DetectorError("Tracking needs the 'lapx' package. Run: pip install lapx "
                                        "(or untick 'Estimate distinct tracks').") from exc
            else:
                results = self.model.predict(frame, **kwargs)
        return extract_detections(results[0])

    def warmup(self) -> None:
        self.infer(np.zeros((320, 320, 3), dtype=np.uint8), conf=0.25, imgsz=320)


def load_engine(model_path: Path | str, device: str = "cpu") -> YoloEngine:
    path = Path(model_path)
    if not path.is_file():
        raise ModelNotFoundError(
            f"Model file not found: {path}. Put your trained weights there "
            "or set the HELMET_MODEL_PATH environment variable."
        )
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] == "ultralytics":
            raise DetectorError("The 'ultralytics' package is not installed. Run: pip install -r requirements.txt") from exc
        raise DetectorError(f"Ultralytics could not start because a package it needs is missing ({exc}). "
                            "Run: pip install -r requirements.txt") from exc
    except Exception as exc:
        # On Windows a broken PyTorch install raises OSError (e.g. "[WinError 126] Error loading c10.dll"),
        # which is not an ImportError. Report it instead of letting it crash the server.
        raise DetectorError(f"Ultralytics or PyTorch is installed but cannot start ({type(exc).__name__}: {exc}). "
                            "Reinstall PyTorch for your Python version, then run: pip install -r requirements.txt") from exc
    try:
        model = YOLO(str(path))
    except Exception as exc:
        raise DetectorError(f"Could not load model '{path.name}': {exc}") from exc
    validate_class_names(model.names)
    try:
        engine = YoloEngine(model, resolve_device(device), path)
        engine.warmup()
    except DetectorError:
        raise
    except Exception as exc:
        raise DetectorError(f"The model file loaded but failed a test inference ({type(exc).__name__}: {exc}). "
                            "This usually means the PyTorch and torchvision versions do not match.") from exc
    return engine


# --------------------------------------------------------------------------- filtering
def filter_detections(dets: list[Detection], frame_w: int, frame_h: int,
                      settings: ProcessingSettings) -> tuple[list[Detection], list[tuple[Detection, str]]]:
    """Drop WithHelmet/WithoutHelmet boxes that are not plausible head-sized boxes.

    These are heuristics aimed at the failure you observed (a 'WithoutHelmet' box drawn
    around a whole scooter or several people). They are NOT ground truth and have not
    been validated against labelled data. Suppressed boxes are counted and reported.

    A helmet-class box is suppressed when it
      * covers more than ``max_helmet_box_area_frac`` of the frame, or
      * has an extreme aspect ratio (> ``max_helmet_aspect``), or
      * encloses the centres of two or more other, clearly smaller helmet-class boxes.
    """
    frame_area = float(frame_w * frame_h)
    helmets = [d for d in dets if d.cls_id in HELMET_CLASSES]
    kept: list[Detection] = []
    suppressed: list[tuple[Detection, str]] = []
    for d in dets:
        if d.cls_id not in HELMET_CLASSES:
            kept.append(d)
            continue
        x1, y1, x2, y2 = d.xyxy
        w, h = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
        area = w * h
        reason = None
        if area / frame_area > settings.max_helmet_box_area_frac:
            reason = "oversized"
        elif max(w / h, h / w) > settings.max_helmet_aspect:
            reason = "extreme_aspect_ratio"
        else:
            inside = 0
            for o in helmets:
                if o is d:
                    continue
                ox1, oy1, ox2, oy2 = o.xyxy
                cx, cy = (ox1 + ox2) / 2, (oy1 + oy2) / 2
                small = (ox2 - ox1) * (oy2 - oy1) < 0.5 * area
                if small and x1 <= cx <= x2 and y1 <= cy <= y2:
                    inside += 1
            if inside >= 2:
                reason = "encloses_multiple_heads"
        if reason:
            suppressed.append((d, reason))
        else:
            kept.append(d)
    return kept, suppressed


# --------------------------------------------------------------------------- counting
class DetectionCounter:
    """Aggregates per-frame detections. Reports frame-level numbers only."""

    def __init__(self) -> None:
        self.frames_inferred = 0
        self.suppressed: Counter = Counter()
        self._total = Counter()
        self._frames_with = Counter()
        self._max_in_frame = Counter()
        self._conf_sum = Counter()
        self._tracks: dict[int, set] = {c: set() for c in EXPECTED_CLASSES}

    def add_frame(self, dets: list[Detection], suppressed: list[tuple[Detection, str]] = ()) -> None:
        self.frames_inferred += 1
        for _, reason in suppressed:
            self.suppressed[reason] += 1
        per_class = Counter(d.cls_id for d in dets)
        for d in dets:
            self._total[d.cls_id] += 1
            self._conf_sum[d.cls_id] += d.conf
            if d.track_id is not None:
                self._tracks[d.cls_id].add(d.track_id)
        for cid, n in per_class.items():
            self._frames_with[cid] += 1
            self._max_in_frame[cid] = max(self._max_in_frame[cid], n)

    def summary(self, tracking: bool = False) -> dict:
        classes = []
        for cid, name in EXPECTED_CLASSES.items():
            total = self._total[cid]
            classes.append({
                "class_id": cid,
                "name": name,
                "detections": total,
                "frames_with_detection": self._frames_with[cid],
                "max_in_one_frame": self._max_in_frame[cid],
                "mean_confidence": round(self._conf_sum[cid] / total, 3) if total else None,
                "distinct_track_ids": len(self._tracks[cid]) if tracking else None,
            })
        return {
            "frames_inferred": self.frames_inferred,
            "classes": classes,
            "suppressed_helmet_boxes": dict(self.suppressed),
        }


class _FlaggedStore:
    """Keeps the highest-confidence WithoutHelmet/TripleRiding frames, spaced apart in time."""

    def __init__(self, max_items: int, min_gap_frames: int) -> None:
        self.max_items, self.min_gap = max_items, max(min_gap_frames, 1)
        self.items: list[dict] = []

    def offer(self, frame_idx: int, score: float, make_item: Callable[[], dict]) -> None:
        if self.max_items <= 0:
            return
        near = [it for it in self.items if abs(it["frame"] - frame_idx) < self.min_gap]
        if near:
            if score <= max(it["confidence"] for it in near):
                return
            for it in near:
                self.items.remove(it)
        elif len(self.items) >= self.max_items:
            worst = min(self.items, key=lambda it: it["confidence"])
            if score <= worst["confidence"]:
                return
            self.items.remove(worst)
        item = make_item()
        item.update(frame=frame_idx, confidence=score)
        self.items.append(item)


# --------------------------------------------------------------------------- drawing
def draw_detections(frame: np.ndarray, dets: list[Detection]) -> np.ndarray:
    scale = max(frame.shape[1] / 1280.0, 0.6)
    thick = max(int(round(2 * scale)), 1)
    font_scale = 0.55 * scale
    for d in dets:
        x1, y1, x2, y2 = (int(round(v)) for v in d.xyxy)
        color = CLASS_COLORS_BGR[d.cls_id]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
        label = f"{d.cls_name} {d.conf:.2f}" + (f" #{d.track_id}" if d.track_id is not None else "")
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        top = max(y1 - th - base - 4, 0)
        cv2.rectangle(frame, (x1, top), (x1 + tw + 6, top + th + base + 4), color, -1)
        cv2.putText(frame, label, (x1 + 3, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_diagnostic(frame: np.ndarray, kept: list[Detection], suppressed: list[tuple[Detection, str]]) -> np.ndarray:
    """Kept boxes in class colours; boxes removed by the sanity filter in grey with the reason."""
    draw_detections(frame, kept)
    scale = max(frame.shape[1] / 1280.0, 0.6)
    thick = max(int(round(2 * scale)), 1)
    for d, reason in suppressed:
        x1, y1, x2, y2 = (int(round(v)) for v in d.xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (130, 130, 130), thick, cv2.LINE_AA)
        label = f"removed ({reason.replace('_', ' ')}): {d.cls_name} {d.conf:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, 1)
        ty = min(y2 + th + base + 4, frame.shape[0] - 2)
        cv2.rectangle(frame, (x1, ty - th - base - 2), (x1 + tw + 6, ty + 2), (60, 60, 60), -1)
        cv2.putText(frame, label, (x1 + 3, ty - base), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def _det_dict(d: Detection, reason: Optional[str] = None) -> dict:
    out = {"class": d.cls_name, "conf": round(d.conf, 4), "box": [round(v, 1) for v in d.xyxy]}
    if d.track_id is not None:
        out["track_id"] = d.track_id
    if reason:
        out["removed_because"] = reason
    return out


def _jpeg(img: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()


# --------------------------------------------------------------------------- video helpers
def probe_video(path: Path | str) -> dict:
    path = Path(path)
    if not path.is_file():
        raise VideoNotFoundError(f"Video file not found: {path.name}")
    if path.stat().st_size == 0:
        raise InvalidVideoError("The video file is empty (0 bytes).")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise InvalidVideoError("The file could not be opened as a video. It may be corrupted or use an unsupported codec.")
        ok, first = cap.read()
        if not ok or first is None:
            raise EmptyVideoError("The video contains no readable frames.")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or first.shape[1]
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or first.shape[0]
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    warnings = []
    if not math.isfinite(fps) or fps <= 0 or fps > 240:
        warnings.append("Frame rate could not be read; assumed 25 FPS.")
        fps = 25.0
    return {
        "width": width, "height": height, "fps": round(fps, 3),
        "frame_count": count if count > 0 else None,
        "duration_s": round(count / fps, 2) if count > 0 else None,
        "warnings": warnings,
    }


def _output_size(w: int, h: int, max_w: int) -> tuple[int, int]:
    scale = 1.0 if not max_w else min(1.0, max_w / float(w))
    ow, oh = int(round(w * scale)), int(round(h * scale))
    return max(ow - ow % 2, 2), max(oh - oh % 2, 2)   # H.264 needs even dimensions


def find_ffmpeg() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _transcode_h264(src: Path, dst: Path) -> bool:
    """Re-encode to browser-playable H.264. Returns False if ffmpeg is missing or fails."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return False
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-an", "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=3600)
        return dst.is_file() and dst.stat().st_size > 0
    except Exception as exc:
        log.warning("H.264 transcode failed (%s); keeping the mp4v file.", exc)
        dst.unlink(missing_ok=True)
        return False


def _count_frames(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return 0
        n = 0
        while cap.grab():
            n += 1
        return n
    finally:
        cap.release()


# --------------------------------------------------------------------------- main pipeline
def process_video(engine, input_path, output_path, settings: Optional[ProcessingSettings] = None, *,
                  progress_cb: Optional[Callable[[int, Optional[int]], None]] = None,
                  cancel_event: Optional[threading.Event] = None,
                  flagged_dir: Optional[Path] = None,
                  diagnostics_dir: Optional[Path] = None) -> dict:
    """Run detection over a video and write an annotated MP4. Returns a result dict.

    ``engine`` needs an ``infer(frame, conf, imgsz, track)`` method (see YoloEngine) and,
    for tracking, ``reset_tracking()``.
    """
    settings = settings or ProcessingSettings()
    settings.validate()
    input_path, output_path = Path(input_path), Path(output_path)
    info = probe_video(input_path)
    out_w, out_h = _output_size(info["width"], info["height"], settings.max_output_width)
    fps = info["fps"]
    total = info["frame_count"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output_path.with_name(output_path.stem + ".raw.mp4")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise InvalidVideoError("The file could not be opened as a video.")
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise VideoWriterError("Could not create the output video file (MP4 writer failed to open).")

    if settings.track and hasattr(engine, "reset_tracking"):
        engine.reset_tracking()

    counter = DetectionCounter()
    store = _FlaggedStore(settings.flagged_frames_max, int(settings.flagged_min_gap_seconds * fps))
    last: list[Detection] = []
    idx = written = 0
    infer_time = 0.0
    diag_on = bool(settings.diagnostics and diagnostics_dir is not None)
    diag_fh = None
    diag_samples: list[dict] = []
    diag_frames_with_removed = 0
    if diag_on:
        diagnostics_dir = Path(diagnostics_dir)
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        diag_fh = (diagnostics_dir / "diagnostics.jsonl").open("w", encoding="utf-8")
    loop_start = time.perf_counter()
    try:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise ProcessingCancelled("Processing was cancelled.")
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame.shape[1] != out_w or frame.shape[0] != out_h:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            inferred_now = idx % settings.frame_stride == 0
            flag_best = clean = None
            if inferred_now:
                t0 = time.perf_counter()
                raw = engine.infer(frame, settings.conf, settings.imgsz, settings.track)
                infer_time += time.perf_counter() - t0
                if settings.sanity_filter:
                    last, suppressed = filter_detections(raw, out_w, out_h, settings)
                else:
                    last, suppressed = raw, []
                counter.add_frame(last, suppressed)
                if diag_on:
                    diag_fh.write(json.dumps({
                        "frame": idx, "time_s": round(idx / fps, 3),
                        "raw": [_det_dict(d) for d in raw],
                        "kept": [_det_dict(d) for d in last],
                        "removed": [_det_dict(d, r) for d, r in suppressed]}) + "\n")
                    if suppressed:
                        diag_frames_with_removed += 1
                        gap_ok = all(abs(it["frame"] - idx) >= max(int(fps), 1) for it in diag_samples)
                        if gap_ok and len(diag_samples) < settings.diagnostic_samples_max:
                            diag_samples.append({
                                "frame": idx, "time_s": round(idx / fps, 2),
                                "reasons": sorted({r for _, r in suppressed}),
                                "jpeg": _jpeg(draw_diagnostic(frame.copy(), last, suppressed))})
                flagged = [d for d in last if d.cls_id in FLAG_CLASSES]
                if flagged:
                    flag_best = max(flagged, key=lambda d: d.conf)
                    clean = frame.copy()          # the unannotated frame, for the review panel
            draw_detections(frame, last)
            writer.write(frame)
            written += 1
            if flag_best is not None:
                store.offer(idx, flag_best.conf, lambda b=flag_best, f=frame, c=clean: {
                    "label": b.cls_name, "jpeg": _jpeg(f), "raw_jpeg": _jpeg(c)})
            idx += 1
            if progress_cb and (idx % 5 == 0 or (total and idx == total)):
                progress_cb(idx, total)
    except BaseException:
        cap.release()
        writer.release()
        if diag_fh:
            diag_fh.close()
        raw_path.unlink(missing_ok=True)
        raise
    loop_time = time.perf_counter() - loop_start
    cap.release()
    writer.release()
    if diag_fh:
        diag_fh.close()

    if written == 0:
        raw_path.unlink(missing_ok=True)
        raise EmptyVideoError("No frames could be decoded from the video.")
    if progress_cb:
        progress_cb(written, total or written)

    t_tr = time.perf_counter()
    h264 = _transcode_h264(raw_path, output_path)
    transcode_time = time.perf_counter() - t_tr
    if h264:
        raw_path.unlink(missing_ok=True)
    else:
        output_path.unlink(missing_ok=True)
        raw_path.replace(output_path)

    out_frames = _count_frames(output_path)
    if out_frames == 0:
        raise VideoWriterError("The output video was written but could not be re-opened.")

    flagged_meta = []
    for i, item in enumerate(sorted(store.items, key=lambda it: it["frame"])):
        entry = {"frame": item["frame"], "time_s": round(item["frame"] / fps, 2),
                 "label": item["label"], "confidence": round(item["confidence"], 3),
                 "file": None, "raw_file": None}
        if flagged_dir is not None:
            flagged_dir = Path(flagged_dir)
            flagged_dir.mkdir(parents=True, exist_ok=True)
            fname, rname = f"flagged_{i:02d}.jpg", f"flagged_{i:02d}_raw.jpg"
            (flagged_dir / fname).write_bytes(item["jpeg"])
            (flagged_dir / rname).write_bytes(item["raw_jpeg"])
            entry["file"], entry["raw_file"] = fname, rname
        flagged_meta.append(entry)

    diagnostics = None
    if diag_on:
        samples_meta = []
        for i, it in enumerate(sorted(diag_samples, key=lambda x: x["frame"])):
            name = f"suppressed_{i:02d}.jpg"
            (diagnostics_dir / name).write_bytes(it["jpeg"])
            samples_meta.append({"file": name, "frame": it["frame"], "time_s": it["time_s"], "reasons": it["reasons"]})
        diagnostics = {"jsonl": "diagnostics.jsonl", "frames_with_removed_boxes": diag_frames_with_removed,
                       "frames_logged": counter.frames_inferred, "samples": samples_meta}

    n_inf = counter.frames_inferred
    stats = {
        "device": getattr(engine, "device", "unknown"),
        "loop_s": round(loop_time, 2),
        "inference_s": round(infer_time, 2),
        "transcode_s": round(transcode_time, 2),
        "mean_inference_ms": round(infer_time / n_inf * 1000, 1) if n_inf else None,
        "processing_fps": round(written / loop_time, 2) if loop_time > 0 else None,
    }
    warnings = list(info["warnings"])
    if total and written != total:
        warnings.append(f"Decoded {written} frames but the container reported {total}.")
    return {
        "input": {k: info[k] for k in ("width", "height", "fps", "frame_count", "duration_s")},
        "output": {"path": str(output_path), "frames": out_frames, "width": out_w, "height": out_h,
                   "fps": fps, "h264": h264, "browser_playable": h264},
        "frames_read": written,
        "frame_stride": settings.frame_stride,
        "tracking": settings.track,
        "counts": counter.summary(tracking=settings.track),
        "stats": stats,
        "diagnostics": diagnostics,
        "flagged_for_review": flagged_meta,
        "warnings": warnings,
    }
