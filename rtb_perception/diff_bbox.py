from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

Bbox = Tuple[int, int, int, int]


def extract_diff_bboxes(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    threshold: int = 25,
    min_area: int = 100,
    kernel_size: int = 3,
) -> List[Bbox]:
    diff = cv2.absdiff(prev_frame, curr_frame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes: List[Bbox] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < min_area:
            continue
        bboxes.append((int(x), int(y), int(x + w), int(y + h)))
    return bboxes
