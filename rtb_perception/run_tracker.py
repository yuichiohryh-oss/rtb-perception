from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

from .diff_bbox import compute_roi_bounds, extract_diff_bboxes
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
    parser.add_argument("--diff-threshold", type=int, default=25, help="Diff threshold")
    parser.add_argument(
        "--blur", type=int, default=0, help="Gaussian blur kernel size (0 disables)"
    )
    parser.add_argument(
        "--diff-step",
        type=int,
        default=1,
        help="Frame step for diff (1 compares to previous frame)",
    )
    parser.add_argument("--kernel-size", type=int, default=3, help="Morphology kernel size")
    parser.add_argument("--min-area", type=int, default=100, help="Min bbox area")
    parser.add_argument("--roi-top", type=float, default=0.14, help="ROI top ratio")
    parser.add_argument("--roi-bottom", type=float, default=0.74, help="ROI bottom ratio")
    parser.add_argument("--roi-left", type=float, default=0.0, help="ROI left ratio")
    parser.add_argument("--roi-right", type=float, default=1.0, help="ROI right ratio")
    parser.add_argument(
        "--side-split",
        type=float,
        default=0.50,
        help="Board split ratio to infer enemy/friendly side",
    )
    parser.add_argument("--iou-thresh", type=float, default=0.3, help="IoU threshold")
    parser.add_argument(
        "--confirm-frames", type=int, default=2, help="Frames to confirm spawn"
    )
    parser.add_argument("--max-missed", type=int, default=5, help="Max missed frames")
    parser.add_argument(
        "--kind-window",
        type=int,
        default=6,
        help="Frames to accumulate movement for kind_guess",
    )
    parser.add_argument(
        "--kind-move-thresh",
        type=float,
        default=10.0,
        help="Movement threshold to infer area_spell vs unit",
    )
    return parser.parse_args()


def prepare_frame_pair(frame_buffer: deque, diff_step: int):
    if diff_step < 1:
        raise ValueError("diff_step must be >= 1")
    if len(frame_buffer) <= diff_step:
        return None
    return frame_buffer[0], frame_buffer[-1]


def run() -> int:
    args = parse_args()
    if args.diff_step < 1:
        raise ValueError("diff_step must be >= 1")
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
    tracker = UnitTracker(
        iou_thresh=args.iou_thresh,
        confirm_frames=args.confirm_frames,
        max_missed=args.max_missed,
        kind_window=args.kind_window,
        kind_move_thresh=args.kind_move_thresh,
    )
    events_path = out_dir / "events.jsonl"

    frame_buffer = deque(maxlen=args.diff_step + 1)
    frame_index = args.start

    with events_path.open("w", encoding="utf-8") as handle:
        while True:
            if args.end is not None and frame_index >= args.end:
                break
            ok, frame = cap.read()
            if not ok:
                break

            frame_buffer.append(frame)
            pair = prepare_frame_pair(frame_buffer, args.diff_step)
            if pair is None:
                frame_index += 1
                continue

            prev_frame, curr_frame = pair
            diff_bboxes = extract_diff_bboxes(
                prev_frame,
                curr_frame,
                threshold=args.diff_threshold,
                min_area=args.min_area,
                kernel_size=args.kernel_size,
                blur_ksize=args.blur,
                roi_top=args.roi_top,
                roi_bottom=args.roi_bottom,
                roi_left=args.roi_left,
                roi_right=args.roi_right,
            )
            time_sec = frame_index / fps if fps and fps > 0 else None
            split_y = int(curr_frame.shape[0] * args.side_split)
            events = tracker.update(frame_index, diff_bboxes, time_sec, split_y=split_y)
            write_events_jsonl(handle, events)

            if args.debug:
                roi_rect = compute_roi_bounds(
                    curr_frame.shape,
                    roi_top=args.roi_top,
                    roi_bottom=args.roi_bottom,
                    roi_left=args.roi_left,
                    roi_right=args.roi_right,
                )
                debug_img = draw_debug_frame(
                    curr_frame,
                    tracker.get_tracks(),
                    events,
                    tracker.get_candidates(),
                    diff_bboxes=diff_bboxes,
                    roi_rect=roi_rect,
                )
                debug_path = debug_dir / f"frame_{frame_index:06d}.jpg"
                cv2.imwrite(str(debug_path), debug_img)

            frame_index += 1

    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
