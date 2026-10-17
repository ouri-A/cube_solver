import json
import random

import numpy as np
import pytest

import sim_view
from cube_model import FACES, SOLVED, apply_moves, random_moves
from sample import lab_to_bgr

# One LAB colour per face, far apart.
LAB = {"U": [95, 0, 0], "R": [45, 65, 45], "F": [55, -60, 40],
       "D": [90, -5, 85], "L": [65, 45, 70], "B": [35, 20, -65]}


def fake_scan(state: str) -> dict:
    """A scan.json whose every sticker reads exactly its letter's colour."""
    faces = {f: {"lab_mean": [LAB[state[9 * k + i]] for i in range(9)]} for k, f in enumerate(FACES)}
    return {"state": state, "faces": faces, "solution": "R U"}


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
    state, colours, title, solution = sim_view.load(str(path))
    assert state == SOLVED
    assert set(colours) == set(FACES)
    assert title == "2026-10-04_141240"
    assert solution == "R U"


def test_load_state_string_uses_standard_colours():
    state, colours, _, solution = sim_view.load(SOLVED.lower())
    assert state == SOLVED
    assert colours is sim_view.STANDARD
    assert solution == ""


@pytest.mark.parametrize("bad", ["U" * 53, "X" * 54, ""])
def test_load_rejects_a_bad_state(bad):
    with pytest.raises(ValueError):
        sim_view.load(bad)


def test_load_scan_without_solution_is_an_error(tmp_path):
    scan = fake_scan(SOLVED) | {"solution": None, "error": "parity"}
    path = tmp_path / "scan.json"
    path.write_text(json.dumps(scan))
    with pytest.raises(ValueError, match="parity"):
        sim_view.load(str(path))


def test_state_string_gets_a_kociemba_solution():
    state = apply_moves(SOLVED, "R U F' L2")
    _, _, _, solution = sim_view.load(state)
    assert apply_moves(state, solution) == SOLVED


def test_inverse_undoes():
    moves = "R U2 F' D L' B2"
    assert sim_view.inverse(moves) == "B2 L D' F U2 R'"
    assert apply_moves(apply_moves(SOLVED, moves), sim_view.inverse(moves)) == SOLVED


@pytest.mark.parametrize("face", FACES)
def test_clockwise_is_a_rotation(face):
    for q in (1, 2, 3):
        R = sim_view.clockwise(face, q)
        assert (R @ R.T == np.eye(3)).all() and round(np.linalg.det(R)) == 1
    assert (sim_view.clockwise(face, 4) == np.eye(3)).all()


# Solutions kociemba gave for real scans on 2026-10-04.
SAVED = [
    "U L2 F' D' R U2 L D R' U R F2 U' L2 F2 L2 D2 L2 D L2 D'",
    "U F U' F B U L B' D L' D' R B2 L2 U' B2 L2 D R2 F2 L2 B2",
    "R L U2 R L' B2 U2 R2 F2 L2 D2 L2 F2",
    "B D B U2 D L U2 F D B' U' R2 D' F2 U' L2 U L2 U2 R2",
]


@pytest.mark.parametrize("moves", SAVED + [random_moves(200, random.Random(7))])
def test_pieces_match_cube_model_after_every_move(moves):
    """The poses the viewer animates to give the same stickers as cube_model, move by move."""
    start = apply_moves(SOLVED, sim_view.inverse(SAVED[0]))      # any scrambled start
    pieces, want = sim_view.Pieces(), start
    for m in moves.split():
        pieces.turn(m)
        want = apply_moves(want, m)
        assert pieces.state(start) == want, m


def test_a_turn_moves_nine_cubies():
    for f in FACES:
        assert len(sim_view.Pieces().turn(f)) == 9
