import numpy as np
import pytest

from cube_solver import guide_anim as ga
from cube_solver.cube_model import _AXES


def facing_camera(M):
    """The face whose outward normal points at the camera (+z) under pose M."""
    return max("URFDLB", key=lambda f: (M @ np.array(_AXES[f][0], float))[2])


def facing_up(M):
    return max("URFDLB", key=lambda f: (M @ np.array(_AXES[f][0], float))[1])


@pytest.mark.parametrize("face", list(ga.ORDER))
def test_each_animation_ends_with_that_face_towards_the_camera(face):
    end = ga.pose_at(face, ga.HOLD_START + ga.MOVE * len(ga.MOVES[face]) + 0.01)
    assert facing_camera(end) == face


@pytest.mark.parametrize("face", "RBL")
def test_side_faces_keep_U_on_top(face):
    end = ga.pose_at(face, ga.HOLD_START + ga.MOVE + 0.01)
    assert facing_up(end) == "U"


def test_turn_left_moves_the_front_to_the_left_of_the_screen():
    mid = ga.pose_at("R", ga.HOLD_START + ga.MOVE / 2)
    assert (mid @ np.array([0, 0, 1.0]))[0] < 0          # F's centre has moved to screen left


def test_render_size_and_mirror():
    known = {"F": [(0, 0, 255)] * 9}
    img = ga.render("R", known, 0.0, size=86)
    assert img.shape == (86, 86, 3) and img.any()
    assert np.array_equal(ga.render("R", known, 0.0, mirror=True, size=86), img[:, ::-1])
