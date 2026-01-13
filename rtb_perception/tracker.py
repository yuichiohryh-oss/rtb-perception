from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .matching import Bbox, Match, greedy_match


@dataclass
class Track:
    track_id: int
    bbox: Bbox
    last_seen_frame: int
    age: int = 0
    missed_frames: int = 0


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
    meta: Optional[dict] = None


class UnitTracker:
    def __init__(
        self,
        iou_thresh: float = 0.3,
        confirm_frames: int = 2,
        max_missed: int = 5,
    ) -> None:
        self.iou_thresh = iou_thresh
        self.confirm_frames = confirm_frames
        self.max_missed = max_missed
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
        )
        self._next_id += 1
        self.tracks[track.track_id] = track
        return track

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
                )
            )

        for track in track_list:
            if track.track_id in matched_tracks:
                continue
            track.missed_frames += 1
            track.age += 1
            if track.missed_frames > self.max_missed:
                events.append(
                    Event(
                        event="disappear",
                        frame=frame_index,
                        t=time_sec,
                        track_id=track.track_id,
                        bbox=track.bbox,
                        age=track.age,
                        missed=track.missed_frames,
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
