import numpy as np
import pytest

import sim_view
from colours import face_colours
from cube_model import FACES
from sample import lab_to_bgr

LAB = {"U": [95, 0, 0], "R": [45, 65, 45], "F": [55, -60, 40],
       "D": [90, -5, 85], "L": [65, 45, 70], "B": [35, 20, -65]}
STATE = "DLLDUBLRLUUDURRLRDBFFUFLUBBBDDLDUFLRRFURLFRBRFDBDBBFFU"


def readings(state: str, noise: float = 0.0, seed: int = 0) -> dict[str, np.ndarray]:
    """In-memory LAB means, (9, 3) per face, every sticker reading its letter's colour."""
    rng = np.random.default_rng(seed)
    return {f: np.array([LAB[state[9 * k + i]] for i in range(9)], float) + rng.normal(0, noise, (9, 3))
            for k, f in enumerate(FACES)}


def test_each_letter_gets_its_read_colour():
    colours = face_colours(readings(STATE), STATE)
    for f in FACES:
        b, g, r = lab_to_bgr(LAB[f])
        assert colours[f] == pytest.approx((r / 255, g / 255, b / 255))


def test_colour_is_the_mean_over_that_letters_stickers():
    lab = readings(STATE, noise=3.0, seed=1)
    flat = np.concatenate([lab[f] for f in FACES])
    letters = np.array(list(STATE))
    colours = face_colours(lab, STATE)
    for f in FACES:
        b, g, r = lab_to_bgr(flat[letters == f].mean(axis=0))
        assert colours[f] == pytest.approx((r / 255, g / 255, b / 255))


def test_viewer_and_scan_agree_on_a_saved_scan():
    lab = readings(STATE, noise=3.0, seed=2)
    saved = {"state": STATE, "faces": {f: {"lab_mean": lab[f].round(2).tolist()} for f in FACES}}
    assert sim_view.scan_colours(saved) == pytest.approx(face_colours(lab, STATE), abs=1 / 255)
