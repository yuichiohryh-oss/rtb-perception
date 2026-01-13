from rtb_perception.io import event_to_dict
from rtb_perception.tracker import Event


def test_event_to_dict_omits_none_optionals():
    event = Event(
        event="disappear",
        frame=5,
        t=None,
        track_id=3,
        bbox=(1, 2, 3, 4),
        iou=None,
        age=None,
        missed=None,
    )
    data = event_to_dict(event)
    assert data["event"] == "disappear"
    assert data["t"] is None
    assert "iou" not in data
    assert "age" not in data
    assert "missed" not in data
    assert "center" not in data
    assert "side" not in data
    assert "kind_guess" not in data
    assert "meta" not in data
