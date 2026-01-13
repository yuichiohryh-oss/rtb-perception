from rtb_perception.diff_bbox import apply_roi_offset, normalize_blur_ksize


def test_apply_roi_offset():
    assert apply_roi_offset((1, 2, 3, 4), 10, 20) == (11, 22, 13, 24)


def test_normalize_blur_ksize_even_to_odd():
    assert normalize_blur_ksize(4) == 5
    assert normalize_blur_ksize(5) == 5
    assert normalize_blur_ksize(0) == 0
    assert normalize_blur_ksize(-1) == 0
