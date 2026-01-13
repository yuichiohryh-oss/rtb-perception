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
        "iou": event.iou,
        "age": event.age,
        "missed": event.missed,
    }
    if event.meta is not None:
        data["meta"] = event.meta
    return data


def write_events_jsonl(handle: IO[str], events: Iterable[Event]) -> None:
    for event in events:
        handle.write(json.dumps(event_to_dict(event), ensure_ascii=False) + "\n")
