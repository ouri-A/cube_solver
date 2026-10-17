import random

import kociemba
import pytest

from cube_model import FACES, SOLVED, apply_moves, check, random_moves


@pytest.mark.parametrize("face", FACES)
def test_four_quarter_turns_are_identity(face):
    assert apply_moves(SOLVED, " ".join([face] * 4)) == SOLVED
    s = apply_moves(SOLVED, "R U F' L2 D B")
    assert apply_moves(apply_moves(s, face), face + "'") == s


@pytest.mark.parametrize("face", FACES)
def test_single_turn_matches_kociemba(face):
    # kociemba undoes a single clockwise turn with the inverse turn
    assert kociemba.solve(apply_moves(SOLVED, face)) == face + "'"


def test_kociemba_round_trip_on_random_scrambles():
    rng = random.Random(0)
    for _ in range(50):
        state = apply_moves(SOLVED, random_moves(25, rng))
        assert check(state).ok
        assert apply_moves(state, kociemba.solve(state)) == SOLVED


def _swap(s, i, j):
    s = list(s)
    s[i], s[j] = s[j], s[i]
    return "".join(s)


def test_check_catches_each_illegal_kind():
    s = apply_moves(SOLVED, "R U F' L2 D B")
    # twist one corner in place (URF: U9 R1 F3 -> cycle the three stickers)
    t = list(s)
    t[8], t[9], t[20] = s[20], s[8], s[9]
    assert any("twist" in p for p in check("".join(t)).problems)
    # flip one edge in place (UR: U6 R2)
    assert any("flipped" in p for p in check(_swap(s, 5, 10)).problems)
    # swap two whole edges (UR <-> UF) without touching corners
    u = _swap(_swap(s, 5, 7), 10, 19)
    assert any("parity" in p for p in check(u).problems)
    # a colour count off
    assert not check(_swap(s, 0, 4)).ok
