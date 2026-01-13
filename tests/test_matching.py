from rtb_perception.matching import greedy_match, iou


def test_iou_basic():
    b1 = (0, 0, 10, 10)
    b2 = (5, 5, 15, 15)
    score = iou(b1, b2)
    assert score == 25 / 175


def test_greedy_match_prefers_max_iou():
    a = [(0, 0, 10, 10), (20, 20, 30, 30)]
    b = [(1, 1, 9, 9), (21, 21, 29, 29)]
    matches = greedy_match(a, b, 0.1)
    assert len(matches) == 2
    assert {m.idx_a for m in matches} == {0, 1}
    assert {m.idx_b for m in matches} == {0, 1}
