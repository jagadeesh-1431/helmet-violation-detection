from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import cleanup
from app.main import create_app
from tests.test_api import _upload, _wait


def _run_job(client, video, **extra):
    vid = _upload(client, video).json()["video_id"]
    r = client.post("/api/jobs", json={"video_id": vid, "filename": "clip.mp4", **extra})
    assert r.status_code == 202, r.text
    job = _wait(client, r.json()["job_id"])
    assert job["status"] == "completed", job
    return job


# ------------------------------------------------------------------ human review
def test_review_decisions_are_saved_separately_from_model_output(tmp_path, fake_engine, short_video):
    app = create_app(engine=fake_engine, upload_dir=tmp_path / "up", results_dir=tmp_path / "res")
    with TestClient(app) as c:
        job = _run_job(c, short_video)
        jid = job["job_id"]
        summary = tmp_path / "res" / jid / "summary.json"
        before = summary.read_bytes()
        frame = job["result"]["flagged_for_review"][0]["file"]

        assert c.get(f"/api/jobs/{jid}/reviews").json()["reviews"] == {}
        r = c.put(f"/api/jobs/{jid}/reviews/{frame}", json={"decision": "confirmed", "note": "no helmet, clear view"})
        assert r.status_code == 200 and r.json()["decision"] == "confirmed"
        got = c.get(f"/api/jobs/{jid}/reviews").json()["reviews"][frame]
        assert got["decision"] == "confirmed" and got["note"] == "no helmet, clear view"

        assert summary.read_bytes() == before, "model output must not change when a reviewer decides"
        assert (tmp_path / "res" / jid / "reviews.json").is_file()
        assert "reviews" not in json.loads(before)

        c.put(f"/api/jobs/{jid}/reviews/{frame}", json={"decision": "rejected", "note": "was a hat"})
        assert c.get(f"/api/jobs/{jid}/reviews").json()["reviews"][frame]["decision"] == "rejected"

    # survives a restart
    app2 = create_app(engine=fake_engine, upload_dir=tmp_path / "up", results_dir=tmp_path / "res")
    with TestClient(app2) as c2:
        assert c2.get(f"/api/jobs/{jid}/reviews").json()["reviews"][frame]["note"] == "was a hat"


def test_review_validation(tmp_path, fake_engine, short_video):
    app = create_app(engine=fake_engine, upload_dir=tmp_path / "up", results_dir=tmp_path / "res")
    with TestClient(app) as c:
        job = _run_job(c, short_video)
        jid, frame = job["job_id"], job["result"]["flagged_for_review"][0]["file"]
        put = lambda name, body: c.put(f"/api/jobs/{jid}/reviews/{name}", json=body)
        assert put(frame, {"decision": "maybe"}).status_code == 422
        assert put(frame, {"decision": "confirmed", "note": "x" * 501}).status_code == 422
        assert put("flagged_99.jpg", {"decision": "confirmed"}).status_code == 404       # not in this result
        assert put("flagged_00_raw.jpg", {"decision": "confirmed"}).status_code == 404   # only main frames
        assert put("..%2f..%2fsummary.json", {"decision": "confirmed"}).status_code in (404, 422)
        assert c.put("/api/jobs/" + "a" * 32 + "/reviews/" + frame, json={"decision": "confirmed"}).status_code == 404


def test_review_requires_finished_job(tmp_path, short_video):
    class Slow:
        device = "cpu"; gate = threading.Event()
        def infer(self, *a, **k):
            self.gate.wait(10); return []
    eng = Slow()
    app = create_app(engine=eng, upload_dir=tmp_path / "up", results_dir=tmp_path / "res")
    with TestClient(app) as c:
        vid = _upload(c, short_video).json()["video_id"]
        jid = c.post("/api/jobs", json={"video_id": vid}).json()["job_id"]
        assert c.get(f"/api/jobs/{jid}/reviews").status_code == 409
        assert c.put(f"/api/jobs/{jid}/reviews/flagged_00.jpg", json={"decision": "confirmed"}).status_code == 409
        c.post(f"/api/jobs/{jid}/cancel"); eng.gate.set(); _wait(c, jid)


def test_review_csv_export_and_formula_injection(tmp_path, fake_engine, short_video):
    import csv, io
    app = create_app(engine=fake_engine, upload_dir=tmp_path / "up", results_dir=tmp_path / "res")
    with TestClient(app) as c:
        job = _run_job(c, short_video)
        jid, items = job["job_id"], job["result"]["flagged_for_review"]
        c.put(f"/api/jobs/{jid}/reviews/{items[0]['file']}", json={"decision": "confirmed", "note": "=HYPERLINK(\"x\")"})
        r = c.get(f"/api/jobs/{jid}/reviews.csv")
        assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
        rows = list(csv.DictReader(io.StringIO(r.text)))
        assert len(rows) == len(items)
        first = next(x for x in rows if x["frame_image"] == items[0]["file"])
        assert first["reviewer_decision"] == "confirmed" and first["reviewer_note"].startswith("'=")
        assert first["model_label"] == items[0]["label"] and first["source_video"] == "clip.mp4"
        assert all(x["reviewer_decision"] in ("confirmed", "unreviewed") for x in rows)


def test_original_frames_served_and_stored_result_not_mutated(tmp_path, fake_engine, short_video):
    app = create_app(engine=fake_engine, upload_dir=tmp_path / "up", results_dir=tmp_path / "res")
    with TestClient(app) as c:
        job = _run_job(c, short_video)
        item = job["result"]["flagged_for_review"][0]
        a = c.get(item["url"]); b = c.get(item["raw_url"])
        assert a.status_code == b.status_code == 200 and a.content != b.content
        stored = app.state.jobs.get(job["job_id"]).result["flagged_for_review"][0]
        assert "url" not in stored and "raw_url" not in stored
        assert c.get(f"/api/results/{job['job_id']}/flagged/flagged_00_raw.jpg.exe").status_code == 404


# ------------------------------------------------------------------ diagnostics and options via the API
def test_diagnostics_via_api(tmp_path, fake_engine, short_video):
    app = create_app(engine=fake_engine, upload_dir=tmp_path / "up", results_dir=tmp_path / "res")
    with TestClient(app) as c:
        off = _run_job(c, short_video)
        assert off["result"]["diagnostics"] is None
        fake_engine.calls = 0            # the scripted oversized box fires on call 7 of each run
        on = _run_job(c, short_video, diagnostics=True)
        d = on["result"]["diagnostics"]
        assert d["frames_with_removed_boxes"] == 1 and d["samples"][0]["reasons"] == ["oversized"]
        assert c.get(d["samples"][0]["url"]).headers["content-type"] == "image/jpeg"
        j = c.get(d["jsonl_url"])
        assert j.status_code == 200 and "attachment" in j.headers["content-disposition"]
        assert len(j.text.strip().splitlines()) == 20
        assert c.get(f"/api/results/{on['job_id']}/diagnostics/..%2fsummary.json").status_code == 404
        assert c.get(f"/api/results/{on['job_id']}/diagnostics/other.txt").status_code == 404
        assert on["settings"]["diagnostics"] is True


def test_output_width_option_via_api(tmp_path, fake_engine, short_video):
    app = create_app(engine=fake_engine, upload_dir=tmp_path / "up", results_dir=tmp_path / "res")
    with TestClient(app) as c:
        vid = _upload(c, short_video).json()["video_id"]
        assert c.post("/api/jobs", json={"video_id": vid, "output_width": 777}).status_code == 422
        job = _run_job(c, short_video, output_width=0)
        assert job["settings"]["output_width"] == 0 and job["result"]["output"]["width"] == 320


def test_stats_reach_the_api(client, short_video):
    job = _run_job(client, short_video)
    st = job["result"]["stats"]
    assert st["processing_fps"] > 0 and st["mean_inference_ms"] is not None


# ------------------------------------------------------------------ failed jobs leave nothing behind
def test_failed_and_cancelled_jobs_remove_their_output_folder(tmp_path, short_video):
    class Boom:
        device = "cpu"
        def infer(self, *a, **k): raise RuntimeError("x")
    app = create_app(engine=Boom(), upload_dir=tmp_path / "up", results_dir=tmp_path / "res")
    with TestClient(app) as c:
        vid = _upload(c, short_video).json()["video_id"]
        job = _wait(c, c.post("/api/jobs", json={"video_id": vid}).json()["job_id"])
        assert job["status"] == "failed"
        assert not (tmp_path / "res" / job["job_id"]).exists()
        assert c.get(f"/api/jobs/{job['job_id']}").json()["status"] == "failed"   # still reported from memory


# ------------------------------------------------------------------ retention and cleanup
def _age(path: Path, hours: float):
    t = time.time() - hours * 3600
    os.utime(path, (t, t))


def test_cleanup_only_touches_files_the_app_created(tmp_path):
    up, res = tmp_path / "up", tmp_path / "res"
    up.mkdir(); res.mkdir()
    old_upload = up / ("a" * 32 + ".mp4"); old_upload.write_bytes(b"x"); _age(old_upload, 100)
    new_upload = up / ("b" * 32 + ".mp4"); new_upload.write_bytes(b"x")
    foreign_upload = up / "my_own_video.mp4"; foreign_upload.write_bytes(b"x"); _age(foreign_upload, 500)
    old_dir = res / ("c" * 32); old_dir.mkdir(); (old_dir / "annotated.mp4").write_bytes(b"x")
    _age(old_dir / "annotated.mp4", 100); _age(old_dir, 100)
    fresh_dir = res / ("d" * 32); fresh_dir.mkdir(); (fresh_dir / "annotated.mp4").write_bytes(b"x")
    foreign_result = res / "test_detection.mp4"; foreign_result.write_bytes(b"x"); _age(foreign_result, 500)
    foreign_dir = res / "my_experiments"; foreign_dir.mkdir(); _age(foreign_dir, 500)

    expired = cleanup.find_expired(up, res, max_age_hours=24)
    assert sorted(p.name for p in expired) == sorted([old_upload.name, old_dir.name])

    assert cleanup.purge(expired, dry_run=True) and old_upload.exists() and old_dir.exists()
    cleanup.purge(expired, dry_run=False)
    assert not old_upload.exists() and not old_dir.exists()
    for keep in (new_upload, foreign_upload, fresh_dir, foreign_result, foreign_dir):
        assert keep.exists(), keep


def test_purge_refuses_foreign_names(tmp_path):
    f = tmp_path / "important.mp4"; f.write_bytes(b"x")
    assert cleanup.purge([f], dry_run=False) == [] and f.exists()


def test_recent_activity_inside_a_result_folder_keeps_it(tmp_path):
    res = tmp_path / "res"; d = res / ("e" * 32); d.mkdir(parents=True)
    (d / "old.json").write_text("{}"); _age(d / "old.json", 100); _age(d, 100)
    (d / "reviews.json").write_text("{}")             # reviewed recently
    assert cleanup.find_expired(tmp_path / "up", res, 24) == []


def test_startup_removes_stale_partials_and_applies_retention(tmp_path, fake_engine):
    up, res = tmp_path / "up", tmp_path / "res"
    up.mkdir(); res.mkdir()
    stale = up / ("1" * 32 + ".part"); stale.write_bytes(b"x"); _age(stale, 5)
    fresh = up / ("2" * 32 + ".part"); fresh.write_bytes(b"x")
    expired = up / ("3" * 32 + ".mp4"); expired.write_bytes(b"x"); _age(expired, 100)
    foreign = res / "keep_me.mp4"; foreign.write_bytes(b"x"); _age(foreign, 1000)
    with TestClient(create_app(engine=fake_engine, upload_dir=up, results_dir=res, retention_hours=24)):
        pass
    assert not stale.exists() and not expired.exists()
    assert fresh.exists() and foreign.exists()


def test_retention_disabled_by_default(tmp_path, fake_engine):
    up, res = tmp_path / "up", tmp_path / "res"
    up.mkdir()
    old = up / ("4" * 32 + ".mp4"); old.write_bytes(b"x"); _age(old, 10000)
    with TestClient(create_app(engine=fake_engine, upload_dir=up, results_dir=res)):
        pass
    assert old.exists()


def test_cleanup_script_is_dry_run_by_default(tmp_path):
    import subprocess, sys
    up, res = tmp_path / "up", tmp_path / "res"
    up.mkdir(); res.mkdir()
    old = up / ("a" * 32 + ".mp4"); old.write_bytes(b"x" * 10); _age(old, 100)
    mine = res / "test_detection.mp4"; mine.write_bytes(b"x"); _age(mine, 1000)
    root = Path(__file__).resolve().parent.parent
    base = [sys.executable, str(root / "scripts" / "cleanup.py"), "--older-than-hours", "24",
            "--uploads", str(up), "--results", str(res)]
    dry = subprocess.run(base, capture_output=True, text=True)
    assert dry.returncode == 0 and "Would remove" in dry.stdout and old.exists()
    assert "test_detection.mp4" not in dry.stdout
    real = subprocess.run(base + ["--delete"], capture_output=True, text=True)
    assert "Removing" in real.stdout and not old.exists() and mine.exists()
    assert subprocess.run(base[:3] + ["0"], capture_output=True, text=True).returncode == 2
