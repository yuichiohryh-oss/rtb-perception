from __future__ import annotations

import json
from typing import IO, Iterable

from .tracker import Event


def event_to_dict(event: Event) -> dict:
    data = {
        "event": event.event,
        "frame": event.frame,
        "t": event.t,
        "track_id": event.track_id,
        "bbox": list(event.bbox),
        "source": event.source,
    }
    if event.iou is not None:
        data["iou"] = event.iou
    if event.age is not None:
        data["age"] = event.age
    if event.missed is not None:
        data["missed"] = event.missed
    if event.center is not None:
        data["center"] = list(event.center)
    if event.side is not None:
        data["side"] = event.side
    if event.kind_guess is not None:
        data["kind_guess"] = event.kind_guess
    if event.meta is not None:
        data["meta"] = event.meta
    return data


def write_events_jsonl(handle: IO[str], events: Iterable[Event]) -> None:
    for event in events:
        handle.write(json.dumps(event_to_dict(event), ensure_ascii=False) + "\n")
