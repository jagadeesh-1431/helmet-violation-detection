from __future__ import annotations

import re
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

ROOT = Path(__file__).resolve().parent.parent


def _upload(client, path: Path, name: str | None = None):
    with path.open("rb") as fh:
        return client.post("/api/upload", files={"file": (name or path.name, fh, "video/mp4")})


def _wait(client, job_id: str, timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed", "cancelled"):
            return job
        time.sleep(0.1)
    raise AssertionError("job did not finish in time")


# ------------------------------------------------------------------ health / page
def test_health_reports_classes_and_model_state(client):
    h = client.get("/api/health").json()
    assert h["model_loaded"] is True
    assert h["classes"] == {"0": "Plate", "1": "WithHelmet", "2": "WithoutHelmet", "3": "TripleRiding"}
    assert h["device"] == "cpu"


def test_upload_page_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Start detection" in r.text and "/static/upload.js" in r.text
    for asset in ("style.css", "common.js", "upload.js", "results.js"):
        assert client.get(f"/static/{asset}").status_code == 200, asset


def test_results_page_served_and_id_validated(client):
    r = client.get("/results/" + "a" * 32)
    assert r.status_code == 200
    assert "/static/results.js" in r.text and 'data-job-id="' + "a" * 32 + '"' in r.text
    assert "Predictions, not verified violations" in r.text
    assert client.get("/results/not-an-id").status_code == 404
    assert client.get("/results/..%2f..%2fetc").status_code in (404, 422)


# ------------------------------------------------------------------ upload validation
def test_upload_valid_video(client, short_video):
    r = _upload(client, short_video)
    assert r.status_code == 200, r.text
    body = r.json()
    assert re.fullmatch(r"[0-9a-f]{32}", body["video_id"])
    assert body["video"]["width"] == 320 and body["video"]["height"] == 240
    assert client.get(body["preview_url"]).status_code == 200


def test_upload_rejects_unsupported_extension(client, tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("hi")
    r = _upload(client, f)
    assert r.status_code == 415 and "Unsupported file type" in r.json()["detail"]


def test_upload_rejects_fake_video(client, tmp_path):
    f = tmp_path / "fake.mp4"
    f.write_text("definitely not a video")
    r = _upload(client, f)
    assert r.status_code == 400
    assert list((tmp_path / "up").iterdir()) == [], "rejected uploads must not be left on disk"


def test_upload_rejects_empty_file(client, tmp_path):
    f = tmp_path / "empty.mp4"
    f.write_bytes(b"")
    assert _upload(client, f).status_code == 400


def test_upload_rejects_oversize(tmp_path, fake_engine, short_video):
    app = create_app(engine=fake_engine, upload_dir=tmp_path / "up", results_dir=tmp_path / "res",
                     max_upload_mb=0)
    with TestClient(app) as c:
        r = _upload(c, short_video)
        assert r.status_code == 413
        assert list((tmp_path / "up").iterdir()) == []


def test_upload_filename_cannot_escape_upload_dir(client, short_video, tmp_path):
    r = _upload(client, short_video, name="../../evil.mp4")
    assert r.status_code == 200
    assert not (tmp_path.parent / "evil.mp4").exists()


# ------------------------------------------------------------------ jobs
def test_full_flow_upload_process_playback_download(client, short_video, fake_engine):
    vid = _upload(client, short_video).json()["video_id"]
    r = client.post("/api/jobs", json={"video_id": vid, "confidence": 0.3, "imgsz": 640,
                                       "frame_stride": 1, "track": False, "sanity_filter": True})
    assert r.status_code == 202
    job = _wait(client, r.json()["job_id"])
    assert job["status"] == "completed", job
    assert job["progress"] == 1.0
    res = job["result"]
    assert res["frames_read"] == 20
    by = {c["name"]: c for c in res["counts"]["classes"]}
    assert by["Plate"]["detections"] == 20 and by["TripleRiding"]["detections"] == 5
    assert fake_engine.calls == 20

    video = client.get(job["video_url"])
    assert video.status_code == 200 and video.headers["content-type"] == "video/mp4"
    assert len(video.content) > 1000
    rng = client.get(job["video_url"], headers={"Range": "bytes=0-99"})
    assert rng.status_code == 206 and len(rng.content) == 100

    dl = client.get(job["download_url"])
    assert dl.status_code == 200 and "attachment" in dl.headers["content-disposition"]

    flagged = res["flagged_for_review"]
    assert flagged and flagged[0]["url"]
    img = client.get(flagged[0]["url"])
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"


def test_job_validation_errors(client, short_video):
    assert client.post("/api/jobs", json={"video_id": "../etc/passwd"}).status_code == 422
    assert client.post("/api/jobs", json={"video_id": "0" * 32}).status_code == 404
    vid = _upload(client, short_video).json()["video_id"]
    assert client.post("/api/jobs", json={"video_id": vid, "confidence": 5}).status_code == 422
    assert client.post("/api/jobs", json={"video_id": vid, "imgsz": 100}).status_code == 422
    assert client.post("/api/jobs", json={"video_id": vid, "frame_stride": 0}).status_code == 422


def test_unknown_job_and_results(client):
    assert client.get("/api/jobs/" + "a" * 32).status_code == 404
    assert client.get("/api/jobs/not-an-id").status_code == 404
    assert client.get("/api/results/" + "a" * 32 + "/video").status_code == 404


def test_result_video_not_ready_returns_409(tmp_path, short_video):
    import threading

    class SlowEngine:
        device = "cpu"
        gate = threading.Event()
        def infer(self, frame, conf, imgsz, track=False):
            self.gate.wait(10)
            return []

    eng = SlowEngine()
    app = create_app(engine=eng, upload_dir=tmp_path / "up", results_dir=tmp_path / "res")
    with TestClient(app) as c:
        vid = _upload(c, short_video).json()["video_id"]
        job_id = c.post("/api/jobs", json={"video_id": vid}).json()["job_id"]
        assert c.get(f"/api/results/{job_id}/video").status_code == 409
        c.post(f"/api/jobs/{job_id}/cancel")
        eng.gate.set()
        assert _wait(c, job_id)["status"] == "cancelled"


def test_engine_failure_is_reported_as_failed_job(tmp_path, short_video):
    class Boom:
        device = "cpu"
        def infer(self, *a, **k):
            raise RuntimeError("secret internal detail")

    app = create_app(engine=Boom(), upload_dir=tmp_path / "up", results_dir=tmp_path / "res")
    with TestClient(app) as c:
        vid = _upload(c, short_video).json()["video_id"]
        job = _wait(c, c.post("/api/jobs", json={"video_id": vid}).json()["job_id"])
        assert job["status"] == "failed"
        assert "secret internal detail" not in job["error"]


def test_missing_model_gives_clear_503_and_health_message(tmp_path, short_video):
    app = create_app(model_path=tmp_path / "models" / "best.pt",
                     upload_dir=tmp_path / "up", results_dir=tmp_path / "res")
    with TestClient(app) as c:
        h = c.get("/api/health").json()
        assert h["model_loaded"] is False and "not found" in h["model_error"].lower()
        vid = _upload(c, short_video).json()["video_id"]
        r = c.post("/api/jobs", json={"video_id": vid})
        assert r.status_code == 503 and "unavailable" in r.json()["detail"].lower()


# ------------------------------------------------------------------ restart / reload
def test_upload_info_endpoint(client, short_video):
    vid = _upload(client, short_video).json()["video_id"]
    info = client.get(f"/api/uploads/{vid}").json()
    assert info["video"]["width"] == 320 and info["preview_url"].endswith("/video")
    assert client.get("/api/uploads/" + "b" * 32).status_code == 404


def test_job_keeps_filename_and_survives_server_restart(tmp_path, short_video, fake_engine):
    up, res = tmp_path / "up", tmp_path / "res"
    app1 = create_app(engine=fake_engine, upload_dir=up, results_dir=res)
    with TestClient(app1) as c1:
        vid = _upload(c1, short_video).json()["video_id"]
        r = c1.post("/api/jobs", json={"video_id": vid, "confidence": 0.4, "frame_stride": 2,
                                       "filename": "traffic.mp4"})
        before = _wait(c1, r.json()["job_id"])
    assert before["filename"] == "traffic.mp4" and before["status"] == "completed"

    # a brand-new app instance with an empty in-memory job table, same folders
    app2 = create_app(engine=fake_engine, upload_dir=up, results_dir=res)
    with TestClient(app2) as c2:
        after = c2.get(f"/api/jobs/{before['job_id']}").json()
        assert after["status"] == "completed" and after["progress"] == 1.0
        assert after["filename"] == "traffic.mp4"
        assert after["settings"]["confidence"] == 0.4 and after["settings"]["frame_stride"] == 2
        assert after["result"]["counts"] == before["result"]["counts"]
        assert c2.get(after["video_url"]).status_code == 200
        assert c2.get(after["result"]["flagged_for_review"][0]["url"]).status_code == 200


def test_corrupt_summary_is_treated_as_not_found(tmp_path, fake_engine):
    app = create_app(engine=fake_engine, upload_dir=tmp_path / "up", results_dir=tmp_path / "res")
    jid = "c" * 32
    (tmp_path / "res" / jid).mkdir(parents=True)
    (tmp_path / "res" / jid / "summary.json").write_text("{ not json")
    with TestClient(app) as c:
        assert c.get(f"/api/jobs/{jid}").status_code == 404


# ------------------------------------------------------------------ frontend <-> backend contract
STATIC = ROOT / "app" / "static"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _route_patterns(app):
    return [re.compile("^" + re.sub(r"\{[^}]+\}", "[^/]+", r.path) + "$")
            for r in app.routes if getattr(r, "path", "").startswith("/api/")]


def test_scripts_only_call_endpoints_that_exist(client):
    pats = _route_patterns(client.app)
    seen = set()
    for js in ("common.js", "upload.js", "results.js"):
        for u in re.findall(r'["\'](/api/[^"\']*)["\']', _read(STATIC / js)):
            seen.add(u)
            probe = u + ("a" * 32 if u.endswith("/") else "")
            assert any(p.match(probe) for p in pats), f"{js} calls {u!r} but no such route exists"
    assert {"/api/health", "/api/upload", "/api/jobs", "/api/jobs/", "/api/uploads/"} <= seen


def test_every_element_id_used_by_each_script_exists_in_its_page(client):
    pages = {"upload.js": client.get("/").text, "results.js": client.get("/results/" + "a" * 32).text}
    for js, html in pages.items():
        ids = set(re.findall(r'\$\("([\w-]+)"\)', _read(STATIC / js)))
        ids |= set(re.findall(r'\b(?:getElementById)\("([\w-]+)"\)', _read(STATIC / js)))
        assert ids, js
        missing = [i for i in ids if f'id="{i}"' not in html]
        assert not missing, f"{js} references ids missing from its page: {missing}"
    results_html = pages["results.js"]
    for v in ("loading", "processing", "error", "done"):      # built dynamically in results.js
        assert f'id="state-{v}"' in results_html
    assert 'id="model-status"' in pages["upload.js"] and 'id="model-status"' in results_html


def test_upload_page_hands_over_to_results_route(client):
    js = _read(STATIC / "upload.js")
    assert 'location.assign("/results/" + job.job_id)' in js
    assert client.get("/results/" + "d" * 32).status_code == 200
    assert "?video=" in _read(STATIC / "results.js") and "get(\"video\")" in js


def test_job_payload_has_fields_scripts_read(client, short_video):
    vid = _upload(client, short_video).json()["video_id"]
    job = _wait(client, client.post("/api/jobs", json={"video_id": vid}).json()["job_id"])
    for key in ("status", "progress", "frames_done", "frames_total", "elapsed_s", "error", "result",
                "video_url", "download_url", "filename", "video_id", "settings"):
        assert key in job, key
    for key in ("confidence", "imgsz", "frame_stride", "track", "sanity_filter"):
        assert key in job["settings"], key
    for key in ("output", "input", "counts", "flagged_for_review", "warnings", "frame_stride", "frames_read"):
        assert key in job["result"], key
    for key in ("browser_playable", "width", "height"):
        assert key in job["result"]["output"]
    assert "duration_s" in job["result"]["input"]
    for c in job["result"]["counts"]["classes"]:
        for key in ("class_id", "name", "detections", "frames_with_detection", "max_in_one_frame",
                    "mean_confidence", "distinct_track_ids"):
            assert key in c, key


def test_upload_recreates_missing_upload_folder(tmp_path, fake_engine, short_video):
    import shutil
    app = create_app(engine=fake_engine, upload_dir=tmp_path / "up", results_dir=tmp_path / "res")
    with TestClient(app) as c:
        shutil.rmtree(tmp_path / "up")
        r = _upload(c, short_video)
        assert r.status_code == 200, r.text


# ------------------------------------------------------------------ polish found from real use
def test_cards_in_two_column_grids_are_not_pushed_down():
    css = _read(STATIC / "style.css")
    assert ".card + .card" in css                       # stacking margin exists for cards one under another
    assert ".grid-2 > .card, .grid-results > .card { margin-top: 0; }" in css
    assert css.index(".grid-2 > .card, .grid-results > .card") > css.index(".card + .card")   # and wins


def test_windows_disconnect_noise_is_filtered_but_real_errors_are_not():
    import logging
    from app.main import QuietClientDisconnects
    f = QuietClientDisconnects()

    def rec(msg, exc):
        info = (type(exc), exc, None) if exc else None
        return logging.LogRecord("asyncio", logging.ERROR, __file__, 1, msg, (), info)

    noise = rec("Exception in callback _ProactorBasePipeTransport._call_connection_lost(None)", ConnectionResetError(10054))
    assert f.filter(noise) is False
    other_reset = rec("Exception in callback something_else()", ConnectionResetError(10054))
    assert f.filter(other_reset) is True                # a reset elsewhere is still reported
    assert f.filter(rec("Exception in callback _ProactorBasePipeTransport._call_connection_lost(None)", ValueError("x"))) is True
    assert f.filter(rec("plain message", None)) is True


def test_disconnect_filter_is_installed_once(tmp_path, fake_engine):
    import logging
    from app.main import QuietClientDisconnects
    create_app(engine=fake_engine, upload_dir=tmp_path / "u", results_dir=tmp_path / "r")
    create_app(engine=fake_engine, upload_dir=tmp_path / "u", results_dir=tmp_path / "r")
    assert sum(isinstance(x, QuietClientDisconnects) for x in logging.getLogger("asyncio").filters) == 1
