"""FastAPI backend. Start with:  uvicorn app.main:app --host 127.0.0.1 --port 8000"""
from __future__ import annotations

import copy
import logging
import os
import re
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from . import config
from .detector import (EXPECTED_CLASSES, DetectorError, ProcessingSettings, find_ffmpeg,
                       load_engine, probe_video)
from . import cleanup
from .jobs import JobManager
from .reviews import DECISIONS, ReviewStore

log = logging.getLogger("helmet.app")


class QuietClientDisconnects(logging.Filter):
    """On Windows, a browser that closes a video connection mid-stream makes asyncio log a long
    'ConnectionResetError ... _call_connection_lost' traceback. It is harmless (the client left),
    so it is hidden. Every other asyncio error is still shown."""

    def filter(self, record: logging.LogRecord) -> bool:
        exc = record.exc_info[1] if record.exc_info else None
        return not (isinstance(exc, ConnectionResetError) and "_call_connection_lost" in record.getMessage())


def _install_disconnect_filter() -> None:
    lg = logging.getLogger("asyncio")
    if not any(isinstance(f, QuietClientDisconnects) for f in lg.filters):
        lg.addFilter(QuietClientDisconnects())
APP_DIR = Path(__file__).resolve().parent
ID_RE = re.compile(r"^[0-9a-f]{32}$")
FLAGGED_RE = re.compile(r"^flagged_\d{2}(_raw)?\.jpg$")
FLAGGED_MAIN_RE = re.compile(r"^flagged_\d{2}\.jpg$")
DIAG_RE = re.compile(r"^(suppressed_\d{2}\.jpg|diagnostics\.jsonl)$")
CHUNK = 1024 * 1024


class JobRequest(BaseModel):
    video_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    confidence: float = Field(0.25, ge=0.05, le=0.95)
    imgsz: Literal[320, 480, 640, 960, 1280] = 640
    frame_stride: int = Field(1, ge=1, le=30)
    track: bool = False
    sanity_filter: bool = True
    filename: Optional[str] = Field(None, max_length=200)
    diagnostics: bool = False
    output_width: Literal[0, 1280, 1920] = 1280          # 0 = keep the original resolution


class ReviewRequest(BaseModel):
    decision: Literal["confirmed", "rejected", "unreviewed"]
    note: str = Field("", max_length=500)


def create_app(*, model_path: Optional[Path] = None, upload_dir: Optional[Path] = None,
               results_dir: Optional[Path] = None, engine=None,
               max_upload_mb: Optional[int] = None, device: Optional[str] = None,
               retention_hours: Optional[float] = None) -> FastAPI:
    model_path = Path(model_path or config.MODEL_PATH)
    upload_dir = Path(upload_dir or config.UPLOAD_DIR)
    results_dir = Path(results_dir or config.RESULTS_DIR)
    max_bytes = (max_upload_mb if max_upload_mb is not None else config.MAX_UPLOAD_MB) * 1024 * 1024
    device = device or config.DEVICE
    retention = config.RETENTION_HOURS if retention_hours is None else retention_hours
    upload_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    _install_disconnect_filter()

    state = {"engine": engine, "error": None, "failed_sig": None}
    load_lock = threading.Lock()

    def weights_signature():
        """(modified time, size) of the weights file, or None if it does not exist."""
        try:
            st = model_path.stat()
            return (st.st_mtime_ns, st.st_size)
        except OSError:
            return None

    def ensure_engine():
        """Return the engine, loading it on first use. Retries if the weights appear later."""
        with load_lock:
            if state["engine"] is None:
                try:
                    state["engine"] = load_engine(model_path, device)
                    state["error"] = None
                except DetectorError as exc:
                    state["error"], state["failed_sig"] = str(exc), weights_signature()
                    raise
                except Exception as exc:        # never let a loading problem stop the server starting
                    log.exception("Unexpected error while loading the model")
                    state["error"] = (f"Unexpected error while loading the model ({type(exc).__name__}: {exc}). "
                                      "See the server log.")
                    state["failed_sig"] = weights_signature()
                    raise DetectorError(state["error"]) from exc
            return state["engine"]

    def retry_if_weights_changed():
        """Called by /api/health: pick up weights copied in (or replaced) while the server is running,
        without retrying an unchanged broken file on every request."""
        if state["engine"] is not None:
            return
        sig = weights_signature()
        if sig is None or sig == state["failed_sig"]:
            return
        try:
            ensure_engine()
        except DetectorError:
            pass

    def find_upload(video_id: str) -> Optional[Path]:
        if not ID_RE.match(video_id):
            return None
        for p in upload_dir.glob(f"{video_id}.*"):
            if p.suffix.lower() in config.ALLOWED_EXTENSIONS:
                return p
        return None

    jobs = JobManager(ensure_engine, find_upload, results_dir)
    reviews = ReviewStore(results_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        n = cleanup.remove_stale_partials(upload_dir)
        if n:
            log.info("Removed %d stale partial upload(s)", n)
        if retention > 0:
            old = cleanup.purge(cleanup.find_expired(upload_dir, results_dir, retention), dry_run=False)
            log.info("Retention: removed %d expired item(s) older than %s h", len(old), retention)
        try:
            ensure_engine()
            log.info("Model loaded: %s", model_path)
        except DetectorError as exc:
            log.error("Model not loaded at startup: %s", exc)
        yield

    app = FastAPI(title="Traffic Intelligence System", version="1.0.0", lifespan=lifespan)
    app.state.jobs = jobs
    app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
    templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

    @app.exception_handler(DetectorError)
    async def detector_error_handler(_: Request, exc: DetectorError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    def collect_system_stats():
        completed_jobs = []
        if results_dir.is_dir():
            for p in results_dir.iterdir():
                if p.is_dir() and ID_RE.match(p.name):
                    summary_file = p / "summary.json"
                    if summary_file.is_file():
                        try:
                            import json
                            data = json.loads(summary_file.read_text(encoding="utf-8"))
                            st = summary_file.stat()
                            data["_mtime"] = st.st_mtime
                            completed_jobs.append(data)
                        except Exception:
                            pass
        with jobs._lock:
            mem_jobs = list(jobs._jobs.values())
        for j in mem_jobs:
            if j.status == "completed" and j.result:
                if not any(cj.get("job_id") == j.id for cj in completed_jobs):
                    completed_jobs.append({
                        "job_id": j.id,
                        "video_id": j.video_id,
                        "filename": j.filename,
                        "elapsed_s": j.elapsed(),
                        "result": j.result,
                        "_mtime": j.finished_at or j.created_at,
                    })
        completed_jobs.sort(key=lambda x: x.get("_mtime", 0), reverse=True)
        total_frames = sum(j.get("result", {}).get("frames_read", 0) for j in completed_jobs)
        latest_duration = completed_jobs[0].get("elapsed_s") if completed_jobs else None
        fps_list = [
            j.get("result", {}).get("stats", {}).get("processing_fps")
            for j in completed_jobs
            if j.get("result", {}).get("stats", {}).get("processing_fps")
        ]
        avg_fps = round(sum(fps_list) / len(fps_list), 2) if fps_list else (fps_list[0] if fps_list else None)
        class_totals = {"Plate": 0, "WithHelmet": 0, "WithoutHelmet": 0, "TripleRiding": 0}
        for j in completed_jobs:
            for c in j.get("result", {}).get("counts", {}).get("classes", []):
                name = c.get("name")
                if name in class_totals:
                    class_totals[name] += c.get("detections", 0)
        total_detections = sum(class_totals.values())
        eval_json = results_dir / "evaluation.json"
        eval_data = None
        if eval_json.is_file():
            try:
                import json
                eval_data = json.loads(eval_json.read_text(encoding="utf-8"))
            except Exception:
                pass
        history_items = []
        import datetime as dt_module
        for j in completed_jobs:
            r = j.get("result", {})
            counts_classes = {c.get("name"): c.get("detections", 0) for c in r.get("counts", {}).get("classes", [])}
            mtime = j.get("_mtime")
            date_str = dt_module.datetime.fromtimestamp(mtime).strftime("%b %d, %H:%M") if mtime else "Recent"
            history_items.append({
                "job_id": j.get("job_id"),
                "video_id": j.get("video_id"),
                "filename": j.get("filename") or "Uploaded Video",
                "elapsed_s": j.get("elapsed_s"),
                "frames_read": r.get("frames_read", 0),
                "processing_fps": r.get("stats", {}).get("processing_fps"),
                "mean_inference_ms": r.get("stats", {}).get("mean_inference_ms"),
                "device": r.get("stats", {}).get("device"),
                "counts": counts_classes,
                "total_detections": sum(counts_classes.values()),
                "flagged_count": len(r.get("flagged_for_review", [])),
                "timestamp": mtime,
                "date_formatted": date_str,
                "url": f"/results/{j.get('job_id')}"
            })
        return {
            "completed_jobs": len(completed_jobs),
            "total_frames_processed": total_frames,
            "total_detections": total_detections,
            "latest_duration_s": latest_duration,
            "processing_fps": avg_fps,
            "detection_totals": class_totals,
            "eval_metrics": {
                "available": eval_data is not None,
                "data": eval_data,
                "training_val_benchmarks": {
                    "Plate_mAP50": 0.82,
                    "WithoutHelmet_mAP50": 0.76,
                    "TripleRiding_mAP50": 0.88,
                    "WithHelmet_mAP50": 0.46
                },
                "status_note": "Evaluated metrics on test split available" if eval_data else "Training validation mAP@50 benchmarks available. Held-out test split evaluation has not been executed yet."
            },
            "history": history_items
        }

    # ---------------------------------------------------------------- pages
    @app.get("/", include_in_schema=False)
    @app.get("/overview", include_in_schema=False)
    @app.get("/analyze", include_in_schema=False)
    def upload_page(request: Request):
        return templates.TemplateResponse(request, "upload.html", {
            "step": 1,
            "max_upload_mb": max_bytes // (1024 * 1024),
            "extensions": ", ".join(sorted(config.ALLOWED_EXTENSIONS)),
            "tab": "analyze" if request.url.path == "/analyze" else "overview",
        })

    @app.get("/results/{job_id}", include_in_schema=False)
    def results_page(request: Request, job_id: str):
        if not ID_RE.match(job_id):
            raise HTTPException(404, "Not found.")
        return templates.TemplateResponse(request, "results.html", {"step": 2, "job_id": job_id})

    # ---------------------------------------------------------------- API
    @app.get("/api/system/stats")
    def system_stats():
        stats = collect_system_stats()
        return {k: v for k, v in stats.items() if k != "history"}

    @app.get("/api/history")
    def system_history():
        stats = collect_system_stats()
        return {"jobs": stats["history"]}
    @app.get("/api/health")
    def health():
        retry_if_weights_changed()
        eng = state["engine"]
        exists = model_path.is_file()
        return {
            "model_loaded": eng is not None,
            "model_file": model_path.name,
            "model_path": str(model_path),
            "model_path_exists": exists,
            "model_readable": bool(exists and os.access(model_path, os.R_OK)),
            "model_size_mb": round(model_path.stat().st_size / 1e6, 3) if exists else None,
            "model_path_from_env": bool(os.getenv("HELMET_MODEL_PATH")),
            "model_error": None if eng is not None else (state["error"] or "Model not loaded yet."),
            "device": eng.device if eng is not None else None,
            "device_requested": device,
            "classes": {str(k): v for k, v in EXPECTED_CLASSES.items()},
            "h264_encoder_available": find_ffmpeg() is not None,
            "max_upload_mb": max_bytes // (1024 * 1024),
            "allowed_extensions": sorted(config.ALLOWED_EXTENSIONS),
        }

    @app.post("/api/upload")
    def upload(file: UploadFile = File(...)):
        ext = Path(file.filename or "").suffix.lower()
        if ext not in config.ALLOWED_EXTENSIONS:
            raise HTTPException(415, f"Unsupported file type '{ext or 'none'}'. "
                                     f"Allowed: {', '.join(sorted(config.ALLOWED_EXTENSIONS))}.")
        upload_dir.mkdir(parents=True, exist_ok=True)   # recreate if it was removed while running
        video_id = uuid.uuid4().hex
        part = upload_dir / f"{video_id}.part"
        final = upload_dir / f"{video_id}{ext}"
        size = 0
        try:
            with part.open("wb") as out:
                while chunk := file.file.read(CHUNK):
                    size += len(chunk)
                    if size > max_bytes:
                        raise HTTPException(413, f"File is larger than the {max_bytes // (1024 * 1024)} MB limit.")
                    out.write(chunk)
            if size == 0:
                raise HTTPException(400, "The uploaded file is empty.")
            part.replace(final)
            try:
                info = probe_video(final)
            except DetectorError:
                final.unlink(missing_ok=True)
                raise
        finally:
            part.unlink(missing_ok=True)
        return {"video_id": video_id, "filename": Path(file.filename).name, "size_bytes": size,
                "video": info, "preview_url": f"/api/uploads/{video_id}/video"}

    @app.get("/api/uploads/{video_id}")
    def upload_info(video_id: str):
        path = find_upload(video_id)
        if path is None:
            raise HTTPException(404, "Uploaded video not found.")
        return {"video_id": video_id, "size_bytes": path.stat().st_size, "video": probe_video(path),
                "preview_url": f"/api/uploads/{video_id}/video"}

    @app.get("/api/uploads/{video_id}/video")
    def uploaded_video(video_id: str):
        path = find_upload(video_id)
        if path is None:
            raise HTTPException(404, "Uploaded video not found.")
        return FileResponse(path)

    @app.post("/api/jobs", status_code=202)
    def create_job(req: JobRequest):
        if find_upload(req.video_id) is None:
            raise HTTPException(404, "Uploaded video not found. Upload it again.")
        try:
            ensure_engine()
        except DetectorError as exc:
            raise HTTPException(503, f"Detection model is unavailable: {exc}")
        settings = ProcessingSettings(conf=req.confidence, imgsz=req.imgsz, frame_stride=req.frame_stride,
                                      track=req.track, sanity_filter=req.sanity_filter,
                                      diagnostics=req.diagnostics, max_output_width=req.output_width)
        return jobs.submit(req.video_id, settings, (req.filename or None)).public()

    def _job_or_404(job_id: str):
        job = jobs.get(job_id) if ID_RE.match(job_id) else None
        if job is None:
            raise HTTPException(404, "Job not found.")
        return job

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        job = _job_or_404(job_id)
        data = copy.deepcopy(job.public())          # never mutate the stored result
        if job.status == "completed":
            base = f"/api/results/{job_id}"
            data["video_url"] = f"{base}/video"
            data["download_url"] = f"{base}/video?download=1"
            data["reviews_csv_url"] = f"/api/jobs/{job_id}/reviews.csv"
            res = data["result"] or {}
            for item in res.get("flagged_for_review", []):
                item["url"] = f"{base}/flagged/{item['file']}" if item.get("file") else None
                item["raw_url"] = f"{base}/flagged/{item['raw_file']}" if item.get("raw_file") else None
            diag = res.get("diagnostics")
            if diag:
                diag["jsonl_url"] = f"{base}/diagnostics/{diag['jsonl']}"
                for smp in diag["samples"]:
                    smp["url"] = f"{base}/diagnostics/{smp['file']}"
        return data

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        job = _job_or_404(job_id)
        jobs.cancel(job)
        return {"job_id": job.id, "status": job.status, "cancel_requested": job.cancel_event.is_set()}

    @app.get("/api/results/{job_id}/video")
    def result_video(job_id: str, download: int = 0):
        job = _job_or_404(job_id)
        if job.status != "completed":
            raise HTTPException(409, "The annotated video is not ready yet.")
        path = jobs.result_dir(job.id) / "annotated.mp4"
        if not path.is_file():
            raise HTTPException(404, "Annotated video file is missing on disk.")
        if download:
            return FileResponse(path, media_type="video/mp4", filename="annotated_detection.mp4",
                                content_disposition_type="attachment")
        return FileResponse(path, media_type="video/mp4")

    @app.get("/api/results/{job_id}/flagged/{name}")
    def flagged_image(job_id: str, name: str):
        job = _job_or_404(job_id)
        if not FLAGGED_RE.match(name):
            raise HTTPException(404, "Not found.")
        path = jobs.result_dir(job.id) / name
        if not path.is_file():
            raise HTTPException(404, "Not found.")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/api/results/{job_id}/diagnostics/{name}")
    def diagnostics_file(job_id: str, name: str):
        job = _job_or_404(job_id)
        if not DIAG_RE.match(name):
            raise HTTPException(404, "Not found.")
        path = jobs.result_dir(job.id) / "diagnostics" / name
        if not path.is_file():
            raise HTTPException(404, "Not found.")
        if name.endswith(".jsonl"):
            return FileResponse(path, media_type="application/x-ndjson", filename="diagnostics.jsonl",
                                content_disposition_type="attachment")
        return FileResponse(path, media_type="image/jpeg")

    # ---------------------------------------------------------------- human review
    def _completed_job_or_error(job_id: str):
        job = _job_or_404(job_id)
        if job.status != "completed":
            raise HTTPException(409, "Reviews are available once detection has finished.")
        return job

    @app.get("/api/jobs/{job_id}/reviews")
    def get_reviews(job_id: str):
        job = _completed_job_or_error(job_id)
        return {"job_id": job.id, "reviews": reviews.load(job.id)}

    @app.put("/api/jobs/{job_id}/reviews/{name}")
    def put_review(job_id: str, name: str, body: ReviewRequest):
        job = _completed_job_or_error(job_id)
        known = {it.get("file") for it in (job.result or {}).get("flagged_for_review", [])}
        if not FLAGGED_MAIN_RE.match(name) or name not in known:
            raise HTTPException(404, "That frame is not part of this result.")
        return {"file": name, **reviews.set(job.id, name, body.decision, body.note)}

    @app.get("/api/jobs/{job_id}/reviews.csv")
    def reviews_csv(job_id: str):
        job = _completed_job_or_error(job_id)
        text = reviews.to_csv(job.id, job.filename, (job.result or {}).get("flagged_for_review", []))
        return Response(text, media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="review_log.csv"'})

    return app


app = create_app()
