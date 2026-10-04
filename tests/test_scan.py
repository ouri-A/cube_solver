import random

import cv2
import numpy as np

import scan
from cube_model import SOLVED, apply_moves
from sample import _SYMMETRIES

NOMINAL = [(95, 0, 0), (88, -6, 78), (45, 62, 45), (62, 42, 62), (58, -52, 38), (38, 12, -50)]


def lab_bgr(lab):
    px = np.array(lab, np.float32).reshape(1, 1, 3)
    return tuple(float(x) * 255 for x in cv2.cvtColor(px, cv2.COLOR_LAB2BGR)[0, 0])


def paint_face(stickers, colours, H):
    """A dark frame with one face drawn through homography H."""
    img = np.full((480, 640, 3), 30, np.uint8)
    for k, letter in enumerate(stickers):
        r, c = divmod(k, 3)
        quad = np.array([[c + .08, r + .08], [c + .92, r + .08], [c + .92, r + .92], [c + .08, r + .92]])
        p = np.c_[quad, np.ones(4)] @ H.T
        cv2.fillConvexPoly(img, np.rint(p[:, :2] / p[:, 2:]).astype(np.int32), colours[letter])
    return img


def similarity(theta, pitch=60.0):
    t = np.radians(theta)
    c, s = np.cos(t) * pitch, np.sin(t) * pitch
    return np.array([[c, -s, 230], [s, c, 140], [0, 0, 1]])


def test_full_scan_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(scan, "SCAN_DIR", tmp_path)
    rng = random.Random(7)
    moves = "R U F' L2 D B'"
    state = apply_moves(SOLVED, moves)
    colours = dict(zip("URFDLB", (lab_bgr(c) for c in rng.sample(NOMINAL, 6))))
    session = scan.ScanSession()
    for face in scan.ORDER:
        k = "URFDLB".index(face)
        H = similarity(rng.uniform(-30, 30))
        frame = paint_face(state[9 * k:9 * k + 9], colours, H)
        assert session.capture(frame, H @ rng.choice(_SYMMETRIES)) == face
    assert session.next_face is None
    result = scan.finish(session, moves)
    assert result.resolution.state == state
    assert result.wrong == []
    assert apply_moves(state, result.solution) == SOLVED
    assert (result.folder / "scan.json").exists() and (result.folder / "U.png").exists()


def test_undo_and_drawing_do_not_crash():
    session = scan.ScanSession()
    H = similarity(5)
    frame = paint_face("UUUUUUUUU", {"U": (200, 200, 200)}, H)
    session.capture(frame, H)
    session.undo()
    assert session.next_face == "F"
    view = frame.copy()
    scan.draw_strip(view, session)
    scan.draw_prompt(view, session, True, "flash")
