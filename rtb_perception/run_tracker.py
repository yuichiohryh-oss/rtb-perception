from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

from .diff_bbox import extract_diff_bboxes
from .io import write_events_jsonl
from .tracker import UnitTracker
from .visualize import draw_debug_frame

Bbox = Tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run diff-based tracking")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--end", type=int, default=None, help="End frame index (exclusive)")
    parser.add_argument("--debug", action="store_true", help="Save debug images")
    parser.add_argument("--min-area", type=int, default=100, help="Min bbox area")
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    video_path = Path(args.video)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir / "debug"
    if args.debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    fps = cap.get(cv2.CAP_PROP_FPS)
    tracker = UnitTracker()
    events_path = out_dir / "events.jsonl"

    prev_frame = None
    frame_index = args.start

    with events_path.open("w", encoding="utf-8") as handle:
        while True:
            if args.end is not None and frame_index >= args.end:
                break
            ok, frame = cap.read()
            if not ok:
                break

            if prev_frame is None:
                prev_frame = frame
                frame_index += 1
                continue

            diff_bboxes = extract_diff_bboxes(prev_frame, frame, min_area=args.min_area)
            time_sec = frame_index / fps if fps and fps > 0 else None
            events = tracker.update(frame_index, diff_bboxes, time_sec)
            write_events_jsonl(handle, events)

            if args.debug:
                debug_img = draw_debug_frame(
                    frame,
                    tracker.get_tracks(),
                    events,
                    tracker.get_candidates(),
                    diff_bboxes=diff_bboxes,
                )
                debug_path = debug_dir / f"frame_{frame_index:06d}.jpg"
                cv2.imwrite(str(debug_path), debug_img)

            prev_frame = frame
            frame_index += 1

    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
