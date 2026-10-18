from cube_executor.moves import bad_moves


def test_face_turns_pass():
    assert bad_moves(['R', "U'", 'F2', 'D', "L'", 'B2']) == []


def test_anything_else_is_reported_in_order():
    assert bad_moves(['R', 'x', 'U3', "R2'", 'r', '', 'M']) == ['x', 'U3', "R2'", 'r', '', 'M']
