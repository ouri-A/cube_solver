import json

import pytest

import sim_view
from cube_model import FACES, SOLVED
from sample import lab_to_bgr

# One LAB colour per face, far apart.
LAB = {"U": [95, 0, 0], "R": [45, 65, 45], "F": [55, -60, 40],
       "D": [90, -5, 85], "L": [65, 45, 70], "B": [35, 20, -65]}


def fake_scan(state: str) -> dict:
    """A scan.json whose every sticker reads exactly its letter's colour."""
    faces = {f: {"lab_mean": [LAB[state[9 * k + i]] for i in range(9)]} for k, f in enumerate(FACES)}
    return {"state": state, "faces": faces}


def test_26_cubies_around_an_empty_core():
    ps = sim_view.cubie_positions()
    assert len(ps) == 26
    assert (0, 0, 0) not in ps
    assert all(set(p) <= {-1, 0, 1} for p in ps)


def test_to_scene_flips_only_z():
    assert sim_view.to_scene((1, 2, 3)) == (1, 2, -3)


def test_scan_colours_are_the_scanned_colours():
    state = "DLLDUBLRLUUDURRLRDBFFUFLUBBBDDLDUFLRRFURLFRBRFDBDBBFFU"
    colours = sim_view.scan_colours(fake_scan(state))
    for f in FACES:
        b, g, r = lab_to_bgr(LAB[f])
        assert colours[f] == pytest.approx((r / 255, g / 255, b / 255))


def test_load_scan_json(tmp_path):
    path = tmp_path / "2026-10-04_141240" / "scan.json"
    path.parent.mkdir()
    path.write_text(json.dumps(fake_scan(SOLVED)))
    state, colours, title = sim_view.load(str(path))
    assert state == SOLVED
    assert set(colours) == set(FACES)
    assert title == "2026-10-04_141240"


def test_load_state_string_uses_standard_colours():
    state, colours, _ = sim_view.load(SOLVED.lower())
    assert state == SOLVED
    assert colours is sim_view.STANDARD


@pytest.mark.parametrize("bad", ["U" * 53, "X" * 54, ""])
def test_load_rejects_a_bad_state(bad):
    with pytest.raises(ValueError):
        sim_view.load(bad)
