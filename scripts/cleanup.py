"""Remove old uploads and results that THIS APP created. Dry run unless you pass --delete.

    python scripts/cleanup.py --older-than-hours 72            # shows what would be removed
    python scripts/cleanup.py --older-than-hours 72 --delete   # removes it

Only items with a server-generated name (a 32-character hex id) are ever listed or removed:
uploads/<id>.mp4 and results/<id>/ folders. Your own files, for example results/test_detection.mp4 or
videos/, are never touched. A results folder counts as recent if anything in it changed recently,
including a saved review.

To clean automatically when the server starts, set HELMET_RETENTION_HOURS (0 = never, the default).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import cleanup, config  # noqa: E402


def _size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--older-than-hours", type=float, required=True)
    ap.add_argument("--uploads", default=str(config.UPLOAD_DIR))
    ap.add_argument("--results", default=str(config.RESULTS_DIR))
    ap.add_argument("--delete", action="store_true", help="actually delete (default is a dry run)")
    args = ap.parse_args()
    if args.older_than_hours <= 0:
        print("--older-than-hours must be greater than 0")
        return 2

    expired = cleanup.find_expired(Path(args.uploads), Path(args.results), args.older_than_hours)
    if not expired:
        print("Nothing to remove.")
        return 0
    total = sum(_size(p) for p in expired)
    verb = "Removing" if args.delete else "Would remove"
    for p in expired:
        print(f"{verb}: {p}  ({_size(p) / 1e6:.1f} MB)")
    cleanup.purge(expired, dry_run=not args.delete)
    print(f"\n{len(expired)} item(s), {total / 1e6:.1f} MB " + ("removed." if args.delete else "found. Run again with --delete to remove them."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
