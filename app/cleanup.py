"""Retention helpers. Only ever touches files this app created.

Owned items are recognised by their server-generated names: a 32-hex-character id. Anything else
in uploads/ or results/ (for example your own test_detection.mp4) is never listed or deleted.
"""
from __future__ import annotations

import logging
import re
import shutil
import time
from pathlib import Path
from typing import Optional

from . import config

log = logging.getLogger(__name__)
HEX = re.compile(r"^[0-9a-f]{32}$")


def _newest_mtime(path: Path) -> float:
    if path.is_dir():
        times = [p.stat().st_mtime for p in path.rglob("*") if p.exists()]
        return max(times + [path.stat().st_mtime])
    return path.stat().st_mtime


def owned_uploads(upload_dir: Path) -> list[Path]:
    upload_dir = Path(upload_dir)
    if not upload_dir.is_dir():
        return []
    return [p for p in upload_dir.iterdir()
            if p.is_file() and HEX.match(p.stem) and p.suffix.lower() in (config.ALLOWED_EXTENSIONS | {".part"})]


def owned_result_dirs(results_dir: Path) -> list[Path]:
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        return []
    return [p for p in results_dir.iterdir() if p.is_dir() and HEX.match(p.name)]


def find_expired(upload_dir: Path, results_dir: Path, max_age_hours: float,
                 now: Optional[float] = None) -> list[Path]:
    now = time.time() if now is None else now
    cutoff = now - max_age_hours * 3600
    items = owned_uploads(upload_dir) + owned_result_dirs(results_dir)
    return sorted(p for p in items if _newest_mtime(p) < cutoff)


def remove_stale_partials(upload_dir: Path, max_age_s: float = 3600, now: Optional[float] = None) -> int:
    """Delete leftover <id>.part files from interrupted uploads."""
    now = time.time() if now is None else now
    n = 0
    for p in owned_uploads(upload_dir):
        if p.suffix == ".part" and now - p.stat().st_mtime > max_age_s:
            p.unlink(missing_ok=True)
            n += 1
    return n


def purge(paths: list[Path], dry_run: bool = True) -> list[Path]:
    done = []
    for p in paths:
        if not (HEX.match(p.stem if p.is_file() else p.name)):      # defensive: never delete foreign names
            continue
        if not dry_run:
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
        done.append(p)
    return done
