"""In-memory job manager. One worker thread keeps CPU inference from being oversubscribed.

Finished jobs are also written to results/<job_id>/summary.json, so a results page keeps working
after the server restarts (the job is reloaded from disk).
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .detector import DetectorError, ProcessingCancelled, ProcessingSettings, process_video

log = logging.getLogger(__name__)
ID_RE = re.compile(r"^[0-9a-f]{32}$")


def _settings_dict(s: ProcessingSettings) -> dict:
    return {"confidence": s.conf, "imgsz": s.imgsz, "frame_stride": s.frame_stride,
            "track": s.track, "sanity_filter": s.sanity_filter, "diagnostics": s.diagnostics,
            "output_width": s.max_output_width}


@dataclass
class Job:
    id: str
    video_id: str
    settings: ProcessingSettings
    filename: Optional[str] = None
    status: str = "queued"          # queued | processing | completed | failed | cancelled
    frames_done: int = 0
    frames_total: Optional[int] = None
    error: Optional[str] = None
    result: Optional[dict] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    elapsed_saved: Optional[float] = None      # set for jobs reloaded from disk
    cancel_event: threading.Event = field(default_factory=threading.Event)

    @property
    def progress(self) -> float:
        if self.status == "completed":
            return 1.0
        if not self.frames_total:
            return 0.0
        return min(self.frames_done / self.frames_total, 0.99)

    def elapsed(self) -> Optional[float]:
        if self.elapsed_saved is not None:
            return self.elapsed_saved
        if self.started_at:
            return round((self.finished_at or time.time()) - self.started_at, 1)
        return None

    def public(self) -> dict:
        return {
            "job_id": self.id, "video_id": self.video_id, "filename": self.filename,
            "status": self.status, "progress": round(self.progress, 4),
            "frames_done": self.frames_done, "frames_total": self.frames_total,
            "elapsed_s": self.elapsed(), "error": self.error,
            "settings": _settings_dict(self.settings), "result": self.result,
        }


class JobManager:
    def __init__(self, get_engine: Callable[[], object], find_upload: Callable[[str], Optional[Path]],
                 results_dir: Path):
        self._get_engine = get_engine
        self._find_upload = find_upload
        self.results_dir = Path(results_dir)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="detector")

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
        return job or self._load_from_disk(job_id)

    def _load_from_disk(self, job_id: str) -> Optional[Job]:
        if not ID_RE.match(job_id):
            return None
        path = self.results_dir / job_id / "summary.json"
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            s = data["settings"]
            job = Job(id=job_id, video_id=data.get("video_id", ""), filename=data.get("filename"),
                      settings=ProcessingSettings(conf=s["confidence"], imgsz=s["imgsz"],
                                                  frame_stride=s["frame_stride"], track=s["track"],
                                                  sanity_filter=s["sanity_filter"],
                                                  diagnostics=s.get("diagnostics", False),
                                                  max_output_width=s.get("output_width", 1280)),
                      status="completed", result=data["result"], elapsed_saved=data.get("elapsed_s"))
            job.frames_done = job.frames_total = data["result"]["frames_read"]
            return job
        except Exception:
            log.warning("Could not reload job %s from disk", job_id, exc_info=True)
            return None

    def submit(self, video_id: str, settings: ProcessingSettings, filename: Optional[str] = None) -> Job:
        job = Job(id=uuid.uuid4().hex, video_id=video_id, settings=settings, filename=filename)
        with self._lock:
            self._jobs[job.id] = job
        self._pool.submit(self._run, job)
        return job

    def cancel(self, job: Job) -> None:
        if job.status in ("queued", "processing"):
            job.cancel_event.set()

    def result_dir(self, job_id: str) -> Path:
        return self.results_dir / job_id

    def _run(self, job: Job) -> None:
        if job.cancel_event.is_set():
            job.status, job.finished_at = "cancelled", time.time()
            return
        job.status, job.started_at = "processing", time.time()
        out_dir = self.result_dir(job.id)

        def on_progress(done: int, total: Optional[int]) -> None:
            job.frames_done, job.frames_total = done, total

        try:
            engine = self._get_engine()
            src = self._find_upload(job.video_id)
            if src is None:
                raise DetectorError("The uploaded video is no longer available. Please upload it again.")
            out_dir.mkdir(parents=True, exist_ok=True)
            result = process_video(engine, src, out_dir / "annotated.mp4", job.settings,
                                   progress_cb=on_progress, cancel_event=job.cancel_event,
                                   flagged_dir=out_dir, diagnostics_dir=out_dir / "diagnostics")
            job.finished_at = time.time()
            summary = {"job_id": job.id, "video_id": job.video_id, "filename": job.filename,
                       "settings": _settings_dict(job.settings), "elapsed_s": job.elapsed(),
                       "result": result}
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            job.result, job.status = result, "completed"
        except ProcessingCancelled:
            job.status = "cancelled"
        except DetectorError as exc:
            job.status, job.error = "failed", str(exc)
        except Exception as exc:
            log.exception("Unexpected error while processing job %s", job.id)
            job.status = "failed"
            job.error = f"Unexpected error while processing the video ({type(exc).__name__}). See the server log."
        finally:
            job.finished_at = job.finished_at or time.time()
            if job.status != "completed":          # never leave half-written output behind
                shutil.rmtree(out_dir, ignore_errors=True)
