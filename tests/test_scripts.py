from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import download_dataset as dd  # noqa: E402
import audit_model  # noqa: E402,F401


# ------------------------------------------------------------------ ZIP validation
def test_validate_zip_good(tmp_path):
    z = tmp_path / "a.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("x.txt", "hello")
    ok, msg = dd.validate_zip(z)
    assert ok and "Valid ZIP" in msg


def test_validate_zip_missing_empty_html_garbage_truncated(tmp_path):
    assert dd.validate_zip(tmp_path / "none.zip")[0] is False
    e = tmp_path / "e.zip"; e.write_bytes(b"")
    assert "0 bytes" in dd.validate_zip(e)[1]
    h = tmp_path / "h.zip"; h.write_text("<!DOCTYPE html><html>403</html>")
    assert "web page" in dd.validate_zip(h)[1]
    g = tmp_path / "g.zip"; g.write_bytes(b"\x00\x01garbage" * 100)
    assert "not a ZIP" in dd.validate_zip(g)[1]
    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as zf:
        zf.writestr("big.bin", np.random.bytes(50000))
    t = tmp_path / "t.zip"; t.write_bytes(good.read_bytes()[:20000])
    ok, msg = dd.validate_zip(t)
    assert not ok and "truncated" in msg


def test_validate_zip_detects_crc_corruption(tmp_path):
    z = tmp_path / "c.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("data.bin", b"A" * 1000)
    raw = bytearray(z.read_bytes())
    raw[60] ^= 0xFF          # flip a byte inside the stored payload
    z.write_bytes(bytes(raw))
    ok, msg = dd.validate_zip(z)
    assert not ok and "CRC" in msg


def test_safe_extract_blocks_zip_slip(tmp_path):
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../escape.txt", "x")
    with pytest.raises(RuntimeError, match="Unsafe path"):
        dd.safe_extract(z, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_safe_extract_good(tmp_path):
    z = tmp_path / "ok.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("train/images/a.jpg", "x")
    assert dd.safe_extract(z, tmp_path / "out") == 1
    assert (tmp_path / "out" / "train" / "images" / "a.jpg").exists()
    assert (tmp_path / "out" / dd.MARKER).exists()


# ------------------------------------------------------------------ dataset inspection
def _img(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), np.zeros((20, 20, 3), np.uint8))


def test_inspect_dataset_finds_planted_problems(tmp_path):
    ds = tmp_path / "ds"
    (ds / "data.yaml").parent.mkdir(parents=True)
    (ds / "data.yaml").write_text("names: [Plate, WithHelmet, WithoutHelmet, TripleRiding]\n")
    for name, lines in {
        "good": ["0 0.5 0.5 0.1 0.1", "1 0.2 0.2 0.05 0.08"],
        "badclass": ["7 0.5 0.5 0.1 0.1"],
        "outside": ["2 0.95 0.5 0.3 0.1"],
        "malformed": ["2 0.5 0.5 0.1"],
        "zero": ["2 0.5 0.5 0 0.1"],
        "dup": ["1 0.5 0.5 0.1 0.1", "1 0.5 0.5 0.1 0.1"],
        # a big WithoutHelmet box containing two small WithHelmet boxes
        "container": ["2 0.5 0.5 0.6 0.6", "1 0.4 0.4 0.05 0.05", "1 0.6 0.6 0.05 0.05"],
    }.items():
        _img(ds / "train" / "images" / f"{name}.jpg")
        (ds / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (ds / "train" / "labels" / f"{name}.txt").write_text("\n".join(lines) + "\n")
    (ds / "train" / "labels" / "orphan.txt").write_text("0 0.5 0.5 0.1 0.1\n")

    out = tmp_path / "report.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "inspect_dataset.py"), "--root", str(ds),
                        "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    rep = json.loads(out.read_text())
    p = rep["problems"]
    assert p["class_id_out_of_range"]["count"] == 1
    assert p["coordinates_outside_image"]["count"] >= 1
    assert p["malformed_line"]["count"] == 1
    assert p["zero_size_box"]["count"] == 1
    assert p["duplicate_box"]["count"] == 1
    assert p["label_without_image"]["count"] == 1
    assert rep["helmet_boxes_containing_multiple_helmet_boxes"]["count"] == 1
    assert rep["class_mapping_mismatches"] == []
    assert rep["box_size"]["WithoutHelmet"]["frac_over_8pct"] > 0


def test_inspect_dataset_detects_swapped_class_names(tmp_path):
    ds = tmp_path / "ds"
    ds.mkdir()
    (ds / "data.yaml").write_text("names: [Plate, WithoutHelmet, WithHelmet, TripleRiding]\n")
    _img(ds / "train" / "images" / "a.jpg")
    (ds / "train" / "labels").mkdir(parents=True)
    (ds / "train" / "labels" / "a.txt").write_text("1 0.5 0.5 0.1 0.1\n")
    out = tmp_path / "r.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "inspect_dataset.py"), "--root", str(ds),
                        "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0
    assert len(json.loads(out.read_text())["class_mapping_mismatches"]) == 2


# ------------------------------------------------------------------ sample clip
def test_make_sample_clip_does_not_overwrite(tmp_path, short_video):
    out = tmp_path / "clip.mp4"
    cmd = [sys.executable, str(ROOT / "scripts" / "make_sample_clip.py"), str(short_video),
           "--seconds", "1", "--width", "160", "--out", str(out)]
    assert subprocess.run(cmd, capture_output=True).returncode == 0
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 160
    cap.release()
    assert subprocess.run(cmd, capture_output=True).returncode == 1


# ------------------------------------------------------------------ leakage check
def _noise_img(path: Path, seed: int, tweak: int = 0):
    path.parent.mkdir(parents=True, exist_ok=True)
    base = np.random.default_rng(seed).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    img = np.clip(base.astype(int) + tweak, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), img)


def test_inspect_dataset_detects_leakage_between_splits(tmp_path):
    ds = tmp_path / "ds"
    for split in ("train", "valid"):
        (ds / split / "labels").mkdir(parents=True)
    (ds / "data.yaml").write_text("names: [Plate, WithHelmet, WithoutHelmet, TripleRiding]\n")
    # same picture in both splits -> exact duplicate (byte-identical copy)
    _noise_img(ds / "train" / "images" / "a.png", seed=1)
    (ds / "valid" / "images").mkdir(parents=True)
    (ds / "valid" / "images" / "a_copy.png").write_bytes((ds / "train" / "images" / "a.png").read_bytes())
    # re-encoded JPEGs of the same picture, slightly brighter -> near duplicate
    _noise_img(ds / "train" / "images" / "b.png", seed=2)
    _noise_img(ds / "valid" / "images" / "b_bright.png", seed=2, tweak=6)
    # unrelated picture
    _noise_img(ds / "valid" / "images" / "c.png", seed=3)
    for split, names in {"train": ["a", "b"], "valid": ["a_copy", "b_bright", "c"]}.items():
        for n in names:
            (ds / split / "labels" / f"{n}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    out = tmp_path / "r.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "inspect_dataset.py"), "--root", str(ds),
                        "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    lk = json.loads(out.read_text())["leakage"]
    assert lk["images_checked"] == 5
    assert lk["exact_duplicates_across_splits"]["count"] == 1
    near = lk["near_duplicates_across_splits"]
    assert near["count"] == 1 and "b.png" in near["examples"][0]["a"] + near["examples"][0]["b"]
    assert "c.png" not in json.dumps(near) and "c.png" not in json.dumps(lk["exact_duplicates_across_splits"])


def test_inspect_dataset_clean_splits_report_no_leakage_and_flags_unlabelled_images(tmp_path):
    ds = tmp_path / "ds"
    for i, split in enumerate(("train", "valid")):
        (ds / split / "labels").mkdir(parents=True)
        _noise_img(ds / split / "images" / f"x{i}.png", seed=10 + i)
        (ds / split / "labels" / f"x{i}.txt").write_text("1 0.5 0.5 0.1 0.1\n")
    _noise_img(ds / "train" / "images" / "bg.png", seed=99)          # no label file at all
    out = tmp_path / "r.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "inspect_dataset.py"), "--root", str(ds),
                        "--out", str(out)], capture_output=True, text=True)
    rep = json.loads(out.read_text())
    assert rep["leakage"]["exact_duplicates_across_splits"]["count"] == 0
    assert rep["leakage"]["near_duplicates_across_splits"]["count"] == 0
    assert rep["problems"]["image_without_label_file"]["count"] == 1
    skip = subprocess.run([sys.executable, str(ROOT / "scripts" / "inspect_dataset.py"), "--root", str(ds),
                           "--out", str(out), "--no-leakage"], capture_output=True, text=True)
    assert "leakage" not in json.loads(out.read_text())


# ------------------------------------------------------------------ contact sheet
def test_visualize_dataset_selects_large_boxes_and_never_modifies_data(tmp_path):
    ds = tmp_path / "ds"
    (ds / "train" / "labels").mkdir(parents=True)
    for name, line in {"big": "2 0.5 0.5 0.6 0.6", "small": "2 0.5 0.5 0.05 0.05", "other": "1 0.5 0.5 0.7 0.7"}.items():
        _noise_img(ds / "train" / "images" / f"{name}.png", seed=hash(name) % 1000)
        (ds / "train" / "labels" / f"{name}.txt").write_text(line + "\n")
    before = sorted((p.name, p.stat().st_size) for p in ds.rglob("*") if p.is_file())
    out = tmp_path / "sheet.jpg"
    cmd = [sys.executable, str(ROOT / "scripts" / "visualize_dataset.py"), "--root", str(ds),
           "--class", "WithoutHelmet", "--min-area", "0.08", "--out", str(out), "--cell-width", "200", "--cols", "2"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    assert "1 matching images" in r.stdout
    sheet = cv2.imread(str(out))
    assert sheet is not None and sheet.shape[1] == 400
    lst = out.with_suffix(".txt").read_text()
    assert "big.png" in lst and "small.png" not in lst and "other.png" not in lst
    assert before == sorted((p.name, p.stat().st_size) for p in ds.rglob("*") if p.is_file())
    nothing = subprocess.run(cmd[:-6] + ["--class", "TripleRiding", "--out", str(out)], capture_output=True, text=True)
    assert nothing.returncode == 1 and "No images matched" in nothing.stdout
    bad = subprocess.run([sys.executable, str(ROOT / "scripts" / "visualize_dataset.py"), "--root", str(ds),
                          "--class", "Banana"], capture_output=True, text=True)
    assert bad.returncode != 0 and "Unknown class" in (bad.stderr + bad.stdout)


# ------------------------------------------------------------------ audit script: raw vs filtered
def test_audit_model_reports_filter_effect_and_saves_diagnostic_frames(tmp_path, short_video, monkeypatch):
    import types
    import audit_model as am
    from tests.conftest import FakeEngine

    names = {0: "Plate", 1: "WithHelmet", 2: "WithoutHelmet", 3: "TripleRiding"}
    monkeypatch.setattr("ultralytics.YOLO", lambda path: types.SimpleNamespace(names=names))
    monkeypatch.setattr(am, "load_engine", lambda *a, **k: FakeEngine())
    out, diag = tmp_path / "audit.json", tmp_path / "diag"
    monkeypatch.setattr(sys, "argv", ["audit_model.py", "--model", "x.pt", "--video", str(short_video),
                                      "--frames", "20", "--out", str(out), "--diag-dir", str(diag)])
    assert am.main() == 0
    rep = json.loads(out.read_text())
    assert rep["class_mapping_ok"] is True
    fx = rep["filter_effect"]
    assert fx["per_class"]["WithoutHelmet"] == {"raw": 21, "kept": 20, "removed": 1}   # the whole-frame box (call 7)
    assert fx["per_class"]["Plate"]["removed"] == 0
    assert fx["reasons"] == {"oversized": 1} and fx["diagnostic_images"] == 1
    imgs = list(diag.glob("*.jpg"))
    assert len(imgs) == 1 and cv2.imread(str(imgs[0])) is not None


def test_audit_model_flags_class_mapping_mismatch(tmp_path, monkeypatch):
    import types
    import audit_model as am
    swapped = {0: "Plate", 1: "WithoutHelmet", 2: "WithHelmet", 3: "TripleRiding"}
    monkeypatch.setattr("ultralytics.YOLO", lambda path: types.SimpleNamespace(names=swapped))
    out = tmp_path / "a.json"
    monkeypatch.setattr(sys, "argv", ["audit_model.py", "--model", "x.pt", "--out", str(out)])
    assert am.main() == 1
    assert json.loads(out.read_text())["class_mapping_ok"] is False
