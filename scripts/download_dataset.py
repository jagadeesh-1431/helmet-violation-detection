"""Download and validate the Kaggle dataset, or just validate an archive you already have.

Check an existing archive (no Kaggle account needed):
    python scripts/download_dataset.py --validate-only path/to/archive.zip

Download from Kaggle (needs your own Kaggle API credentials, see README):
    pip install kaggle
    python scripts/download_dataset.py --dest data

Credentials are read by the Kaggle library from  %USERPROFILE%\\.kaggle\\kaggle.json  (Windows) or
~/.kaggle/kaggle.json, or from the KAGGLE_USERNAME and KAGGLE_KEY environment variables. This script
never prints, stores or accepts credentials on the command line.

Safety: the archive is fully validated (structure and CRC) before extraction; extraction refuses paths
that would escape the target folder; an already extracted dataset is not extracted again, and nothing
is ever deleted.
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

DATASET_SLUG = "devgurucodes/trafffic-violations-triple-riding-no-helmet-plate"   # spelling as on Kaggle
MARKER = ".extracted_ok"


def validate_zip(path: Path) -> tuple[bool, str]:
    """Return (ok, explanation). Distinguishes the usual reasons an archive 'fails validation'."""
    path = Path(path)
    if not path.is_file():
        return False, f"File not found: {path}"
    size = path.stat().st_size
    if size == 0:
        return False, "The file is 0 bytes: the download or upload did not complete."
    with path.open("rb") as fh:
        head = fh.read(512)
    if not head.startswith(b"PK"):
        if head.lstrip().lower().startswith((b"<!doctype", b"<html", b"{")):
            return False, ("The file is a web page or JSON text, not a ZIP. This usually means the download was "
                           "blocked (login or permission error) and the error page was saved as archive.zip.")
        return False, "The file does not start with a ZIP signature, so it is not a ZIP archive."
    if not zipfile.is_zipfile(path):
        return False, (f"The file starts like a ZIP but its central directory is missing ({size:,} bytes). "
                       "It is almost certainly truncated: the transfer was interrupted. Download it again.")
    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
            if bad:
                return False, f"CRC check failed for '{bad}': the archive is corrupted. Download it again."
            n = len(zf.infolist())
    except zipfile.BadZipFile as exc:
        return False, f"Not a valid ZIP: {exc}"
    return True, f"Valid ZIP with {n} entries ({size / 1e6:.1f} MB)."


def safe_extract(zip_path: Path, dest: Path) -> int:
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            target = (dest / info.filename).resolve()
            if dest != target and dest not in target.parents:
                raise RuntimeError(f"Unsafe path in archive, refusing to extract: {info.filename}")
        zf.extractall(dest)
        count = len(zf.infolist())
    (dest / MARKER).write_text(f"extracted from {zip_path.name}\n")
    return count


def have_credentials() -> bool:
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return True
    cfg = Path(os.getenv("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    return cfg.is_file()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate-only", metavar="ZIP", help="only validate this archive and exit")
    ap.add_argument("--dest", default="data", help="folder for the download (default: data)")
    ap.add_argument("--no-extract", action="store_true")
    args = ap.parse_args()

    if args.validate_only:
        ok, msg = validate_zip(Path(args.validate_only))
        print(("VALID: " if ok else "INVALID: ") + msg)
        return 0 if ok else 1

    dest = Path(args.dest)
    raw = dest / "raw"
    if (raw / MARKER).exists():
        print(f"Dataset already extracted in {raw}. Nothing to do (delete the folder yourself to redo it).")
        return 0
    zip_path = dest / (DATASET_SLUG.split("/")[1] + ".zip")
    if not zip_path.exists():
        if not have_credentials():
            print("Kaggle credentials not found.\n"
                  "  1. Kaggle > Settings > API > 'Create New Token' downloads kaggle.json.\n"
                  "  2. Put it in %USERPROFILE%\\.kaggle\\kaggle.json (Windows) or ~/.kaggle/kaggle.json,\n"
                  "     or set KAGGLE_USERNAME and KAGGLE_KEY. Never commit this file.\n"
                  "  3. Accept the dataset's terms on its Kaggle page if asked, then run this script again.")
            return 3
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError:
            print("The 'kaggle' package is missing. Run: pip install kaggle")
            return 3
        api = KaggleApi()
        api.authenticate()
        dest.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {DATASET_SLUG} ...")
        api.dataset_download_files(DATASET_SLUG, path=str(dest), unzip=False, quiet=False)
        if not zip_path.exists():
            zips = sorted(dest.glob("*.zip"), key=lambda p: p.stat().st_mtime)
            if not zips:
                print("Download finished but no .zip was found in", dest)
                return 1
            zip_path = zips[-1]
    ok, msg = validate_zip(zip_path)
    print(("VALID: " if ok else "INVALID: ") + msg)
    if not ok:
        print("Nothing was extracted or deleted. Fix the cause above and run again.")
        return 1
    if args.no_extract:
        return 0
    n = safe_extract(zip_path, raw)
    print(f"Extracted {n} entries to {raw}")
    print("Next: python scripts/inspect_dataset.py --root", raw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
