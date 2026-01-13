from collections import deque

from rtb_perception.run_tracker import prepare_frame_pair


def test_prepare_frame_pair_uses_diff_step_buffer():
    buffer = deque(maxlen=3)

    buffer.append("f0")
    assert prepare_frame_pair(buffer, diff_step=2) is None

    buffer.append("f1")
    assert prepare_frame_pair(buffer, diff_step=2) is None

    buffer.append("f2")
    assert prepare_frame_pair(buffer, diff_step=2) == ("f0", "f2")

    buffer.append("f3")
    assert prepare_frame_pair(buffer, diff_step=2) == ("f1", "f3")
