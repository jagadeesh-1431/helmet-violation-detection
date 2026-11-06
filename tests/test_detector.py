from __future__ import annotations

import threading

import cv2
import numpy as np
import pytest

from app import detector
from app.detector import (ClassMappingError, Detection, DetectionCounter, EmptyVideoError,
                          InvalidVideoError, ModelNotFoundError, ProcessingCancelled,
                          ProcessingSettings, VideoNotFoundError, VideoWriterError,
                          extract_detections, filter_detections, load_engine, process_video,
                          validate_class_names)

GOOD_NAMES = {0: "Plate", 1: "WithHelmet", 2: "WithoutHelmet", 3: "TripleRiding"}


# ------------------------------------------------------------------ class mapping / model loading
def test_expected_mapping_is_the_four_class_mapping():
    assert detector.EXPECTED_CLASSES == GOOD_NAMES


def test_validate_class_names_accepts_correct_mapping():
    validate_class_names(GOOD_NAMES)
    validate_class_names(["Plate", "WithHelmet", "WithoutHelmet", "TripleRiding"])
    validate_class_names({0: "plate", 1: "with_helmet", 2: "Without Helmet", 3: "triple-riding"})


def test_validate_class_names_rejects_swapped_ids():
    swapped = {0: "Plate", 1: "WithoutHelmet", 2: "WithHelmet", 3: "TripleRiding"}
    with pytest.raises(ClassMappingError, match="class 1"):
        validate_class_names(swapped)


def test_validate_class_names_rejects_wrong_count():
    with pytest.raises(ClassMappingError):
        validate_class_names({i: f"coco{i}" for i in range(80)})


def test_missing_model_file(tmp_path):
    with pytest.raises(ModelNotFoundError, match="not found"):
        load_engine(tmp_path / "missing.pt")


def test_corrupt_model_file(tmp_path):
    pytest.importorskip("ultralytics")
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"this is not a model")
    with pytest.raises(detector.DetectorError):
        load_engine(bad)


def test_resolve_device_defaults_to_cpu():
    assert detector.resolve_device("cpu") == "cpu"
    assert detector.resolve_device("") == "cpu"


# ------------------------------------------------------------------ extract_detections
class _T:
    def __init__(self, a): self.a = np.asarray(a)
    def cpu(self): return self
    def numpy(self): return self.a
    def __len__(self): return len(self.a)


class _Boxes:
    def __init__(self, xyxy, conf, cls, ids=None):
        self.xyxy, self.conf, self.cls = _T(xyxy), _T(conf), _T(cls)
        self.id = _T(ids) if ids is not None else None
    def __len__(self): return len(self.cls)


class _Result:
    def __init__(self, boxes): self.boxes = boxes


def test_extract_detections_maps_ids_to_names():
    res = _Result(_Boxes([[0, 0, 10, 10], [5, 5, 20, 20]], [0.9, 0.6], [2, 3], ids=[7, 8]))
    dets = extract_detections(res)
    assert [(d.cls_name, d.track_id) for d in dets] == [("WithoutHelmet", 7), ("TripleRiding", 8)]


def test_extract_detections_empty_and_unknown_class():
    assert extract_detections(_Result(None)) == []
    assert extract_detections(_Result(_Boxes(np.zeros((0, 4)), [], []))) == []
    assert extract_detections(_Result(_Boxes([[0, 0, 1, 1]], [0.9], [9]))) == []


# ------------------------------------------------------------------ sanity filter
def _d(cid, box, conf=0.8):
    return Detection(cid, detector.EXPECTED_CLASSES[cid], conf, box)


def test_filter_suppresses_oversized_helmet_box_but_keeps_triple_riding():
    s = ProcessingSettings()
    big_without = _d(2, (0, 0, 800, 600))
    big_triple = _d(3, (0, 0, 800, 600))
    kept, sup = filter_detections([big_without, big_triple], 1000, 800, s)
    assert kept == [big_triple]
    assert [r for _, r in sup] == ["oversized"]


def test_filter_suppresses_box_enclosing_multiple_heads():
    s = ProcessingSettings(max_helmet_box_area_frac=0.5)
    outer = _d(2, (100, 100, 300, 260))
    kept, sup = filter_detections([outer, _d(1, (120, 110, 150, 140)), _d(1, (220, 120, 250, 150))], 1000, 1000, s)
    assert outer not in kept and len(kept) == 2
    assert sup[0][1] == "encloses_multiple_heads"


def test_filter_suppresses_extreme_aspect_ratio():
    kept, sup = filter_detections([_d(2, (0, 0, 200, 20))], 1000, 1000, ProcessingSettings())
    assert kept == [] and sup[0][1] == "extreme_aspect_ratio"


def test_filter_keeps_normal_head_boxes_and_plates():
    dets = [_d(1, (10, 10, 40, 45)), _d(2, (100, 100, 130, 135)), _d(0, (0, 0, 200, 20))]
    kept, sup = filter_detections(dets, 1000, 1000, ProcessingSettings())
    assert kept == dets and sup == []


# ------------------------------------------------------------------ counting
def test_counter_aggregates_frame_level_numbers():
    c = DetectionCounter()
    c.add_frame([_d(2, (0, 0, 5, 5), 0.5), _d(2, (9, 9, 14, 14), 0.7), _d(0, (0, 0, 9, 3), 0.9)])
    c.add_frame([])
    c.add_frame([_d(2, (0, 0, 5, 5), 0.9)], suppressed=[(_d(2, (0, 0, 1, 1)), "oversized")])
    s = c.summary()
    by = {row["name"]: row for row in s["classes"]}
    assert s["frames_inferred"] == 3
    assert by["WithoutHelmet"]["detections"] == 3
    assert by["WithoutHelmet"]["frames_with_detection"] == 2
    assert by["WithoutHelmet"]["max_in_one_frame"] == 2
    assert by["WithoutHelmet"]["mean_confidence"] == pytest.approx(0.7, abs=1e-3)
    assert by["Plate"]["detections"] == 1
    assert by["WithHelmet"]["detections"] == 0 and by["WithHelmet"]["mean_confidence"] is None
    assert by["TripleRiding"]["detections"] == 0
    assert by["Plate"]["distinct_track_ids"] is None
    assert s["suppressed_helmet_boxes"] == {"oversized": 1}


def test_counter_distinct_tracks_only_when_tracking():
    c = DetectionCounter()
    for tid in (4, 4, 5):
        c.add_frame([Detection(2, "WithoutHelmet", 0.8, (0, 0, 5, 5), tid)])
    by = {r["name"]: r for r in c.summary(tracking=True)["classes"]}
    assert by["WithoutHelmet"]["detections"] == 3
    assert by["WithoutHelmet"]["distinct_track_ids"] == 2


# ------------------------------------------------------------------ video processing
def test_process_video_missing_file(tmp_path, fake_engine):
    with pytest.raises(VideoNotFoundError):
        process_video(fake_engine, tmp_path / "nope.mp4", tmp_path / "out.mp4")


def test_process_video_zero_byte_file(tmp_path, fake_engine):
    f = tmp_path / "empty.mp4"
    f.write_bytes(b"")
    with pytest.raises(InvalidVideoError):
        process_video(fake_engine, f, tmp_path / "out.mp4")


def test_process_video_not_a_video(tmp_path, fake_engine):
    f = tmp_path / "text.mp4"
    f.write_text("hello, I am not a video")
    with pytest.raises((InvalidVideoError, EmptyVideoError)):
        process_video(fake_engine, f, tmp_path / "out.mp4")


def test_process_video_writer_failure(tmp_path, fake_engine, short_video, monkeypatch):
    class BadWriter:
        def __init__(self, *a, **k): pass
        def isOpened(self): return False
        def release(self): pass
    monkeypatch.setattr(detector.cv2, "VideoWriter", BadWriter)
    with pytest.raises(VideoWriterError):
        process_video(fake_engine, short_video, tmp_path / "out.mp4")
    assert fake_engine.calls == 0


def test_process_video_success_output_reopens_with_same_frame_count(tmp_path, fake_engine, short_video):
    out = tmp_path / "out" / "annotated.mp4"
    flagged = tmp_path / "out"
    result = process_video(fake_engine, short_video, out, ProcessingSettings(), flagged_dir=flagged)
    assert out.is_file() and out.stat().st_size > 0
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened()
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    assert n == 20 == result["output"]["frames"] == result["frames_read"]
    assert not out.with_name("annotated.raw.mp4").exists()
    assert fake_engine.calls == 20
    assert result["counts"]["frames_inferred"] == 20
    assert result["flagged_for_review"], "WithoutHelmet/TripleRiding frames should be queued for review"
    for item in result["flagged_for_review"]:
        assert (flagged / item["file"]).stat().st_size > 0


def test_process_video_counts_match_scripted_engine_and_suppression(tmp_path, fake_engine, short_video):
    result = process_video(fake_engine, short_video, tmp_path / "o.mp4")
    by = {r["name"]: r for r in result["counts"]["classes"]}
    assert by["Plate"]["detections"] == 20
    assert by["WithoutHelmet"]["detections"] == 20          # the whole-frame box on call 7 is excluded
    assert by["TripleRiding"]["detections"] == 5
    assert result["counts"]["suppressed_helmet_boxes"] == {"oversized": 1}


def test_process_video_sanity_filter_can_be_disabled(tmp_path, fake_engine, short_video):
    result = process_video(fake_engine, short_video, tmp_path / "o.mp4", ProcessingSettings(sanity_filter=False))
    by = {r["name"]: r for r in result["counts"]["classes"]}
    assert by["WithoutHelmet"]["detections"] == 21


def test_process_video_frame_stride_reduces_inference_not_output_length(tmp_path, fake_engine, short_video):
    result = process_video(fake_engine, short_video, tmp_path / "o.mp4", ProcessingSettings(frame_stride=4))
    assert fake_engine.calls == 5
    assert result["counts"]["frames_inferred"] == 5
    assert result["output"]["frames"] == 20


def test_process_video_downscales_wide_input(tmp_path, fake_engine, short_video):
    result = process_video(fake_engine, short_video, tmp_path / "o.mp4", ProcessingSettings(max_output_width=320))
    assert result["output"]["width"] == 320
    result = process_video(fake_engine, short_video, tmp_path / "o2.mp4", ProcessingSettings(max_output_width=1920))
    assert result["output"]["width"] == 320, "never upscale"


def test_process_video_progress_and_tracking_reset(tmp_path, fake_engine, short_video):
    seen = []
    process_video(fake_engine, short_video, tmp_path / "o.mp4", ProcessingSettings(track=True),
                  progress_cb=lambda d, t: seen.append((d, t)))
    assert seen[-1][0] == 20
    assert fake_engine.reset_calls == 1


def test_process_video_cancel_cleans_up(tmp_path, fake_engine, short_video):
    ev = threading.Event()
    ev.set()
    out = tmp_path / "o.mp4"
    with pytest.raises(ProcessingCancelled):
        process_video(fake_engine, short_video, out, cancel_event=ev)
    assert not out.exists() and not out.with_name("o.raw.mp4").exists()


def test_settings_validation(tmp_path, fake_engine, short_video):
    with pytest.raises(ValueError):
        process_video(fake_engine, short_video, tmp_path / "o.mp4", ProcessingSettings(conf=1.5))
    with pytest.raises(ValueError):
        process_video(fake_engine, short_video, tmp_path / "o.mp4", ProcessingSettings(frame_stride=0))


# ------------------------------------------------------------------ stats, diagnostics, raw frames
def test_processing_stats_are_recorded(tmp_path, fake_engine, short_video):
    res = process_video(fake_engine, short_video, tmp_path / "o.mp4")
    st = res["stats"]
    assert st["device"] == "cpu" and st["loop_s"] > 0 and st["processing_fps"] > 0
    assert st["mean_inference_ms"] is not None and st["inference_s"] <= st["loop_s"]
    assert st["transcode_s"] >= 0


def test_diagnostics_off_by_default_writes_nothing(tmp_path, fake_engine, short_video):
    d = tmp_path / "diag"
    res = process_video(fake_engine, short_video, tmp_path / "o.mp4", diagnostics_dir=d)
    assert res["diagnostics"] is None and not d.exists()


def test_diagnostics_records_raw_kept_and_removed(tmp_path, fake_engine, short_video):
    import json
    d = tmp_path / "diag"
    res = process_video(fake_engine, short_video, tmp_path / "o.mp4",
                        ProcessingSettings(diagnostics=True), diagnostics_dir=d)
    rows = [json.loads(line) for line in (d / "diagnostics.jsonl").read_text().splitlines()]
    assert len(rows) == 20 == res["diagnostics"]["frames_logged"]
    bad = [r for r in rows if r["removed"]]
    assert len(bad) == 1 and bad[0]["frame"] == 6                      # engine call 7 = frame index 6
    assert bad[0]["removed"][0]["removed_because"] == "oversized"
    assert len(bad[0]["raw"]) == len(bad[0]["kept"]) + 1               # raw keeps what the filter dropped
    assert all(not r["removed"] for r in rows if r["frame"] != 6)
    sample = res["diagnostics"]["samples"][0]
    assert sample["reasons"] == ["oversized"] and (d / sample["file"]).stat().st_size > 0
    assert res["diagnostics"]["frames_with_removed_boxes"] == 1


def test_diagnostics_not_filtered_when_filter_disabled(tmp_path, fake_engine, short_video):
    d = tmp_path / "diag"
    res = process_video(fake_engine, short_video, tmp_path / "o.mp4",
                        ProcessingSettings(diagnostics=True, sanity_filter=False), diagnostics_dir=d)
    assert res["diagnostics"]["frames_with_removed_boxes"] == 0 and res["diagnostics"]["samples"] == []


def test_flagged_frames_have_annotated_and_original_images(tmp_path, fake_engine, short_video):
    out = tmp_path / "res"
    res = process_video(fake_engine, short_video, out / "o.mp4", flagged_dir=out)
    item = res["flagged_for_review"][0]
    ann = cv2.imread(str(out / item["file"]))
    raw = cv2.imread(str(out / item["raw_file"]))
    assert ann is not None and raw is not None and ann.shape == raw.shape
    assert int(np.abs(ann.astype(int) - raw.astype(int)).sum()) > 0, "annotated frame must differ from the original"
    # (x=20, y=10) lies on the Plate box's top edge: plain background in the original (JPEG-tolerant),
    # a strongly coloured pixel in the annotated frame
    assert int(np.abs(raw[10, 20].astype(int) - 60).max()) < 12
    assert int(np.abs(ann[10, 20].astype(int) - 60).max()) > 40


def test_output_width_zero_keeps_original_size(tmp_path, fake_engine, short_video):
    res = process_video(fake_engine, short_video, tmp_path / "o.mp4", ProcessingSettings(max_output_width=0))
    assert (res["output"]["width"], res["output"]["height"]) == (320, 240)
    with pytest.raises(ValueError):
        ProcessingSettings(max_output_width=100).validate()
