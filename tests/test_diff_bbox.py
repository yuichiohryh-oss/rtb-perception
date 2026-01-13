from rtb_perception.diff_bbox import apply_roi_offset


def test_apply_roi_offset():
    assert apply_roi_offset((1, 2, 3, 4), 10, 20) == (11, 22, 13, 24)
