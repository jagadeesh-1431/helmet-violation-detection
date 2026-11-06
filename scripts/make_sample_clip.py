"""Cut a short, smaller clip from a long 4K video for quick testing. Never modifies the original.

    python scripts/make_sample_clip.py "videos/sample.mp4.mp4" --seconds 5 --width 1280
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--out", default="videos/sample_clip.mp4")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        print(f"{out} already exists; choose another --out (this script never overwrites).")
        return 1
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Cannot open {args.video}")
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, args.width / w)
    ow, oh = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start * fps))
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
    if not writer.isOpened():
        print("Could not open the output writer.")
        return 3
    n, limit = 0, int(args.seconds * fps)
    while n < limit:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_AREA))
        n += 1
    cap.release()
    writer.release()
    if n == 0:
        out.unlink(missing_ok=True)
        print("No frames were read.")
        return 4
    print(f"Wrote {n} frames ({n / fps:.1f} s, {ow}x{oh}) to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
