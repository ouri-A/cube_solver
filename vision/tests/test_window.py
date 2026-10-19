import numpy as np

from cube_solver import scan, window
from cube_solver.cube_model import FACES
from cube_solver.sample import FaceReading


def reading(lab):
    lab = np.tile(np.asarray(lab, float), (9, 1))
    return FaceReading(lab_mean=lab, lab_median=lab, spread=np.zeros(9), flat=np.zeros((3, 3, 3), np.uint8))


def test_unread_faces_are_grey():
    assert window.sticker_colours(scan.ScanSession()) == [window.UNREAD] * 54


def test_a_read_face_shows_in_its_measured_colour():
    session = scan.ScanSession()
    session.readings["F"] = reading((55, 75, 60))          # a red
    rgb = window.sticker_colours(session)
    k = FACES.index("F")
    face = rgb[9 * k:9 * k + 9]
    assert len(rgb) == 54 and len(set(face)) == 1
    r, g, b = face[0]
    assert all(0 <= c <= 1 for c in face[0]) and r > 0.6 > max(g, b)
    assert rgb[:9 * k] + rgb[9 * k + 9:] == [window.UNREAD] * 45


def test_the_bottom_bar_names_the_problem():
    assert "sixth face" in window.status_line(None)
    failed = scan.Result(resolution=None, solution=None, error="corner twisted", wrong=None, folder=None)
    assert "corner twisted" in window.status_line(failed)
