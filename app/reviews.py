"""Human review decisions.

Stored in results/<job_id>/reviews.json, SEPARATE from summary.json (the model output), so a
reviewer's decision never overwrites or blends into what the model predicted.
"""
from __future__ import annotations

import csv
import io
import json
import os
import threading
import time
from pathlib import Path

DECISIONS = ("confirmed", "rejected", "unreviewed")
NOTE_MAX = 500


def _csv_safe(value) -> str:
    """Neutralise spreadsheet formula injection in text the reviewer typed."""
    text = "" if value is None else str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


class ReviewStore:
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self._lock = threading.Lock()

    def _path(self, job_id: str) -> Path:
        return self.results_dir / job_id / "reviews.json"

    def load(self, job_id: str) -> dict:
        path = self._path(job_id)
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("reviews", {}) if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def set(self, job_id: str, file: str, decision: str, note: str) -> dict:
        if decision not in DECISIONS:
            raise ValueError("invalid decision")
        note = (note or "").strip()[:NOTE_MAX]
        with self._lock:
            reviews = self.load(job_id)
            entry = {"decision": decision, "note": note, "updated_at": round(time.time(), 1)}
            reviews[file] = entry
            path = self._path(job_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"reviews": reviews}, indent=2), encoding="utf-8")
            os.replace(tmp, path)
        return entry

    def to_csv(self, job_id: str, source_filename: str | None, flagged: list[dict]) -> str:
        reviews = self.load(job_id)
        out = io.StringIO()
        w = csv.writer(out)
        w.writerow(["job_id", "source_video", "frame_image", "time_s", "model_label", "model_confidence",
                    "reviewer_decision", "reviewer_note", "reviewed_at_unix"])
        for it in flagged:
            r = reviews.get(it.get("file") or "", {})
            w.writerow([job_id, _csv_safe(source_filename), it.get("file"), it.get("time_s"), it.get("label"),
                        it.get("confidence"), r.get("decision", "unreviewed"), _csv_safe(r.get("note", "")),
                        r.get("updated_at", "")])
        return out.getvalue()
