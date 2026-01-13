from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .tracker import Candidate, Event, Track

Bbox = Tuple[int, int, int, int]


def _draw_label(img: np.ndarray, text: str, x: int, y: int, color: Tuple[int, int, int]) -> None:
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_debug_frame(
    frame: np.ndarray,
    tracks: Iterable[Track],
    events: List[Event],
    candidates: Iterable[Candidate],
    diff_bboxes: Optional[List[Bbox]] = None,
    roi_rect: Optional[Bbox] = None,
) -> np.ndarray:
    canvas = frame.copy()

    if roi_rect:
        x1, y1, x2, y2 = roi_rect
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (220, 220, 220), 1)

    for cand in candidates:
        x1, y1, x2, y2 = cand.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (180, 180, 180), 1)
        _draw_label(canvas, f"cand s={cand.streak}", x1, max(12, y1 - 4), (180, 180, 180))

    if diff_bboxes:
        for bbox in diff_bboxes:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (200, 200, 0), 1)

    for track in tracks:
        x1, y1, x2, y2 = track.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 0), 2)
        label = f"id={track.track_id} a={track.age} m={track.missed_frames}"
        _draw_label(canvas, label, x1, max(14, y1 - 6), (0, 200, 0))

    y = 18
    for event in events:
        text = f"{event.event} id={event.track_id}"
        _draw_label(canvas, text, 6, y, (0, 255, 255))
        y += 14

    return canvas
