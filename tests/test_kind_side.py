from rtb_perception.tracker import UnitTracker


def test_side_split_enemy_friendly():
    tracker = UnitTracker(confirm_frames=1)
    split_y = 50

    events = tracker.update(0, [(0, 0, 10, 10)], split_y=split_y)
    assert [e.side for e in events] == ["enemy"]

    events = tracker.update(1, [(0, 80, 10, 90)], split_y=split_y)
    assert [e.side for e in events] == ["friendly"]


def test_kind_guess_area_spell():
    tracker = UnitTracker(confirm_frames=1, kind_window=3, kind_move_thresh=2.0)
    bboxes = [(0, 0, 10, 10), (0, 0, 10, 10), (0, 0, 10, 10)]

    tracker.update(0, [bboxes[0]])
    tracker.update(1, [bboxes[1]])
    events = tracker.update(2, [bboxes[2]])

    assert [e.kind_guess for e in events] == ["area_spell"]


def test_kind_guess_unit():
    tracker = UnitTracker(confirm_frames=1, kind_window=3, kind_move_thresh=2.0)
    bboxes = [(0, 0, 10, 10), (0, 5, 10, 15), (0, 10, 10, 20)]

    tracker.update(0, [bboxes[0]])
    tracker.update(1, [bboxes[1]])
    events = tracker.update(2, [bboxes[2]])

    assert [e.kind_guess for e in events] == ["unit"]
