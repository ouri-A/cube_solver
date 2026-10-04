import numpy as np
import pytest

from sample import _SYMMETRIES, read_face, upright


def similarity(theta_deg, pitch=50.0, x0=200.0, y0=150.0):
    t = np.radians(theta_deg)
    c, s = np.cos(t) * pitch, np.sin(t) * pitch
    return np.array([[c, -s, x0], [s, c, y0], [0, 0, 1]])


@pytest.mark.parametrize("theta", [-40, -15, 0, 20, 40])
def test_upright_undoes_any_lattice_symmetry(theta):
    H = similarity(theta)                       # u right, v down, tilted theta
    for S in _SYMMETRIES:                       # the tracker may hand any of the 8 labellings
        assert np.allclose(upright(H @ S), H)


def test_reads_each_cell_in_row_major_order():
    H = similarity(10)
    img = np.zeros((400, 400, 3), np.uint8)
    import cv2
    for k in range(9):                          # paint cell k with blue level 20*k
        r, c = divmod(k, 3)
        quad = np.array([[c, r], [c + 1, r], [c + 1, r + 1], [c, r + 1]], float)
        p = np.c_[quad, np.ones(4)] @ H.T
        cv2.fillConvexPoly(img, np.rint(p[:, :2] / p[:, 2:]).astype(np.int32), (20 * k + 40, 0, 0))
    S = _SYMMETRIES[5]                          # a scrambled lattice labelling
    reading = read_face(img, H @ S)
    blue_order = np.argsort(reading.lab_mean[:, 2])[::-1]   # b channel falls as blue grows
    assert list(blue_order) == list(range(9))
