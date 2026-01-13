from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Optional, Tuple

from .matching import Bbox, Match, greedy_match


@dataclass
class Track:
    track_id: int
    bbox: Bbox
    last_seen_frame: int
    age: int = 0
    missed_frames: int = 0
    last_center: Optional[Tuple[float, float]] = None
    dist_sum: float = 0.0
    kind_guess: str = "unknown"


@dataclass
class Candidate:
    bbox: Bbox
    last_seen_frame: int
    streak: int


@dataclass
class Event:
    event: str
    frame: int
    t: Optional[float]
    track_id: int
    bbox: Bbox
    source: str = "diff"
    iou: Optional[float] = None
    age: Optional[int] = None
    missed: Optional[int] = None
    center: Optional[Tuple[float, float]] = None
    side: Optional[str] = None
    kind_guess: Optional[str] = None
    meta: Optional[dict] = None


class UnitTracker:
    def __init__(
        self,
        iou_thresh: float = 0.3,
        confirm_frames: int = 2,
        max_missed: int = 5,
        kind_window: int = 6,
        kind_move_thresh: float = 10.0,
    ) -> None:
        self.iou_thresh = iou_thresh
        self.confirm_frames = confirm_frames
        self.max_missed = max_missed
        self.kind_window = max(1, kind_window)
        self.kind_move_thresh = kind_move_thresh
        self._next_id = 1
        self.tracks: Dict[int, Track] = {}
        self._candidates: List[Candidate] = []

    def _new_track(self, frame_index: int, bbox: Bbox) -> Track:
        track = Track(
            track_id=self._next_id,
            bbox=bbox,
            last_seen_frame=frame_index,
            age=1,
            missed_frames=0,
            last_center=None,
            dist_sum=0.0,
            kind_guess="unknown",
        )
        self._next_id += 1
        self.tracks[track.track_id] = track
        return track

    @staticmethod
    def _bbox_center(bbox: Bbox) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _infer_side(center: Tuple[float, float], split_y: Optional[int]) -> Optional[str]:
        if split_y is None:
            return None
        return "enemy" if center[1] < split_y else "friendly"

    def _update_track_kind(self, track: Track, center: Tuple[float, float]) -> None:
        if track.last_center is not None and track.age <= self.kind_window:
            dx = center[0] - track.last_center[0]
            dy = center[1] - track.last_center[1]
            track.dist_sum += math.hypot(dx, dy)
        track.last_center = center
        if track.age >= self.kind_window:
            track.kind_guess = (
                "area_spell" if track.dist_sum <= self.kind_move_thresh else "unit"
            )
        else:
            track.kind_guess = "unknown"

    def _refresh_kind_guess(self, track: Track) -> None:
        if track.age >= self.kind_window:
            track.kind_guess = (
                "area_spell" if track.dist_sum <= self.kind_move_thresh else "unit"
            )
        else:
            track.kind_guess = "unknown"

    def _track_match(self, candidates: List[Bbox]) -> List[Match]:
        track_list = list(self.tracks.values())
        track_bboxes = [t.bbox for t in track_list]
        matches = greedy_match(track_bboxes, candidates, self.iou_thresh)
        return matches

    def update(
        self,
        frame_index: int,
        candidates: List[Bbox],
        time_sec: Optional[float] = None,
        split_y: Optional[int] = None,
    ) -> List[Event]:
        events: List[Event] = []
        track_list = list(self.tracks.values())
        track_bboxes = [t.bbox for t in track_list]

        matches = greedy_match(track_bboxes, candidates, self.iou_thresh)
        matched_tracks = set()
        matched_candidates = set()

        for match in matches:
            track = track_list[match.idx_a]
            bbox = candidates[match.idx_b]
            track.bbox = bbox
            track.last_seen_frame = frame_index
            track.missed_frames = 0
            track.age += 1
            center = self._bbox_center(bbox)
            self._update_track_kind(track, center)
            matched_tracks.add(track.track_id)
            matched_candidates.add(match.idx_b)
            events.append(
                Event(
                    event="update",
                    frame=frame_index,
                    t=time_sec,
                    track_id=track.track_id,
                    bbox=bbox,
                    iou=match.iou,
                    age=track.age,
                    missed=track.missed_frames,
                    center=center,
                    side=self._infer_side(center, split_y),
                    kind_guess=track.kind_guess,
                )
            )

        for track in track_list:
            if track.track_id in matched_tracks:
                continue
            track.missed_frames += 1
            track.age += 1
            self._refresh_kind_guess(track)
            if track.missed_frames > self.max_missed:
                center = self._bbox_center(track.bbox)
                events.append(
                    Event(
                        event="disappear",
                        frame=frame_index,
                        t=time_sec,
                        track_id=track.track_id,
                        bbox=track.bbox,
                        age=track.age,
                        missed=track.missed_frames,
                        center=center,
                        side=self._infer_side(center, split_y),
                        kind_guess=track.kind_guess,
                    )
                )
                del self.tracks[track.track_id]

        unmatched_bboxes = [b for i, b in enumerate(candidates) if i not in matched_candidates]

        cand_matches = greedy_match(
            [c.bbox for c in self._candidates],
            unmatched_bboxes,
            self.iou_thresh,
        )
        matched_cands = set()
        matched_unmatched = set()
        spawned_indices = set()

        for match in cand_matches:
            cand = self._candidates[match.idx_a]
            bbox = unmatched_bboxes[match.idx_b]
            cand.bbox = bbox
            cand.last_seen_frame = frame_index
            cand.streak += 1
            matched_cands.add(match.idx_a)
            matched_unmatched.add(match.idx_b)

            if cand.streak >= self.confirm_frames:
                track = self._new_track(frame_index, bbox)
                center = self._bbox_center(bbox)
                self._update_track_kind(track, center)
                events.append(
                    Event(
                        event="spawn",
                        frame=frame_index,
                        t=time_sec,
                        track_id=track.track_id,
                        bbox=bbox,
                        iou=match.iou,
                        age=track.age,
                        missed=track.missed_frames,
                        center=center,
                        side=self._infer_side(center, split_y),
                        kind_guess=track.kind_guess,
                    )
                )
                spawned_indices.add(match.idx_a)

        # keep only candidates seen this frame and not spawned
        kept_candidates: List[Candidate] = []
        for idx, cand in enumerate(self._candidates):
            if idx in spawned_indices:
                continue
            if idx in matched_cands:
                kept_candidates.append(cand)
        self._candidates = kept_candidates

        for i, bbox in enumerate(unmatched_bboxes):
            if i in matched_unmatched:
                continue
            if self.confirm_frames <= 1:
                track = self._new_track(frame_index, bbox)
                center = self._bbox_center(bbox)
                self._update_track_kind(track, center)
                events.append(
                    Event(
                        event="spawn",
                        frame=frame_index,
                        t=time_sec,
                        track_id=track.track_id,
                        bbox=bbox,
                        iou=None,
                        age=track.age,
                        missed=track.missed_frames,
                        center=center,
                        side=self._infer_side(center, split_y),
                        kind_guess=track.kind_guess,
                    )
                )
                continue

            self._candidates.append(
                Candidate(
                    bbox=bbox,
                    last_seen_frame=frame_index,
                    streak=1,
                )
            )

        return events

    def get_tracks(self) -> List[Track]:
        return list(self.tracks.values())

    def get_candidates(self) -> List[Candidate]:
        return list(self._candidates)
