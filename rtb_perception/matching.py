from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

Bbox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class Match:
    idx_a: int
    idx_b: int
    iou: float


def iou(b1: Bbox, b2: Bbox) -> float:
    x_left = max(b1[0], b2[0])
    y_top = max(b1[1], b2[1])
    x_right = min(b1[2], b2[2])
    y_bottom = min(b1[3], b2[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    inter_area = float((x_right - x_left) * (y_bottom - y_top))
    area1 = float((b1[2] - b1[0]) * (b1[3] - b1[1]))
    area2 = float((b2[2] - b2[0]) * (b2[3] - b2[1]))
    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def greedy_match(
    track_bboxes: Sequence[Bbox],
    det_bboxes: Sequence[Bbox],
    iou_thresh: float,
) -> List[Match]:
    matches: List[Match] = []
    candidates: List[Match] = []

    for i, b1 in enumerate(track_bboxes):
        for j, b2 in enumerate(det_bboxes):
            score = iou(b1, b2)
            if score >= iou_thresh:
                candidates.append(Match(i, j, score))

    candidates.sort(key=lambda m: m.iou, reverse=True)
    used_a = set()
    used_b = set()
    for cand in candidates:
        if cand.idx_a in used_a or cand.idx_b in used_b:
            continue
        used_a.add(cand.idx_a)
        used_b.add(cand.idx_b)
        matches.append(cand)

    return matches
