from rtb_perception.tracker import UnitTracker


def test_spawn_confirm_requires_two_frames():
    tracker = UnitTracker(confirm_frames=2, max_missed=5)
    bboxes = [(0, 0, 10, 10)]

    events = tracker.update(0, bboxes)
    assert [e.event for e in events] == []

    events = tracker.update(1, bboxes)
    assert [e.event for e in events] == ["spawn"]
    assert tracker.get_tracks()[0].bbox == bboxes[0]


def test_disappear_after_max_missed():
    tracker = UnitTracker(confirm_frames=1, max_missed=1)
    bboxes = [(0, 0, 10, 10)]

    events = tracker.update(0, bboxes)
    assert [e.event for e in events] == ["spawn"]

    events = tracker.update(1, [])
    assert [e.event for e in events] == []

    events = tracker.update(2, [])
    assert [e.event for e in events] == ["disappear"]
