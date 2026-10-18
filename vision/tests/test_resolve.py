import random

import numpy as np
import pytest

from cube_solver.cube_model import CENTRES, FACES, SOLVED, apply_moves, random_moves
from cube_solver.resolve import resolve

# Rough LAB of the six colours, for code tests only. The real test uses your scans.
NOMINAL = [(95, 0, 0), (88, -6, 78), (45, 62, 45), (62, 42, 62), (58, -52, 38), (38, 12, -50)]


def synthetic(state, rng, noise):
    np_rng = np.random.default_rng(rng.randrange(1 << 30))
    colour = dict(zip(FACES, rng.sample(NOMINAL, 6)))
    lab = np.array([colour[c] for c in state], float) + np_rng.normal(0, noise, (54, 3))
    refs = {f: lab[i].copy() for f, i in zip(FACES, CENTRES)}
    return lab, refs


@pytest.mark.parametrize("method", ["tsp", "assignment"])
def test_recovers_random_cubes(method):
    rng = random.Random(1)
    for _ in range(200):
        state = apply_moves(SOLVED, random_moves(25, rng))
        lab, refs = synthetic(state, rng, noise=6.0)
        res = resolve(lab, refs, method)
        assert res.state == state
        assert res.check.ok


def test_solved_cube():
    lab, refs = synthetic(SOLVED, random.Random(2), noise=3.0)
    assert resolve(lab, refs).state == SOLVED


def test_logo_centre_is_rescued():
    rng = random.Random(3)
    ok = 0
    for _ in range(100):
        state = apply_moves(SOLVED, random_moves(25, rng))
        lab, refs = synthetic(state, rng, noise=5.0)
        f = rng.choice(FACES)
        refs[f] = np.array([30.0, 0.0, 0.0])      # a dark logo swallowed the whole centre patch
        ok += resolve(lab, refs).state == state
    assert ok == 100
