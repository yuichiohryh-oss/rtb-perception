from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

Bbox = Tuple[int, int, int, int]


def compute_roi_bounds(
    frame_shape: Tuple[int, int, int],
    roi_top: float,
    roi_bottom: float,
    roi_left: float,
    roi_right: float,
) -> Bbox:
    height, width = frame_shape[:2]
    x1 = int(width * roi_left)
    x2 = int(width * roi_right)
    y1 = int(height * roi_top)
    y2 = int(height * roi_bottom)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    return x1, y1, x2, y2


def apply_roi_offset(bbox: Bbox, roi_x1: int, roi_y1: int) -> Bbox:
    x1, y1, x2, y2 = bbox
    return x1 + roi_x1, y1 + roi_y1, x2 + roi_x1, y2 + roi_y1


def normalize_blur_ksize(blur_ksize: int) -> int:
    if blur_ksize <= 0:
        return 0
    if blur_ksize % 2 == 0:
        return blur_ksize + 1
    return blur_ksize


def extract_diff_bboxes(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    threshold: int = 25,
    min_area: int = 100,
    kernel_size: int = 3,
    blur_ksize: int = 0,
    roi_top: float = 0.14,
    roi_bottom: float = 0.74,
    roi_left: float = 0.0,
    roi_right: float = 1.0,
) -> List[Bbox]:
    roi_x1, roi_y1, roi_x2, roi_y2 = compute_roi_bounds(
        prev_frame.shape,
        roi_top=roi_top,
        roi_bottom=roi_bottom,
        roi_left=roi_left,
        roi_right=roi_right,
    )
    if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
        return []

    prev_crop = prev_frame[roi_y1:roi_y2, roi_x1:roi_x2]
    curr_crop = curr_frame[roi_y1:roi_y2, roi_x1:roi_x2]
    prev_gray = cv2.cvtColor(prev_crop, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_crop, cv2.COLOR_BGR2GRAY)
    blur_ksize = normalize_blur_ksize(blur_ksize)
    if blur_ksize > 0:
        prev_gray = cv2.GaussianBlur(prev_gray, (blur_ksize, blur_ksize), 0)
        curr_gray = cv2.GaussianBlur(curr_gray, (blur_ksize, blur_ksize), 0)
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

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
        bbox = (int(x), int(y), int(x + w), int(y + h))
        bboxes.append(apply_roi_offset(bbox, roi_x1, roi_y1))
    return bboxes
