import random

import cv2
import numpy as np

from cube_solver import scan
from cube_solver.cube_model import SOLVED, apply_moves
from cube_solver.sample import _SYMMETRIES

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
    for t in (0.0, 1.0, 5.0):
        scan.draw_prompt(view, session, True, "flash", t)


class FakePopen:
    calls = []

    def __init__(self, args):
        FakePopen.calls.append(args)


def test_viewer_opens_on_the_saved_scan(tmp_path, monkeypatch):
    monkeypatch.setattr(scan.subprocess, "Popen", FakePopen)
    FakePopen.calls = []
    result = scan.Result(resolution=None, solution="R U", error=None, wrong=None, folder=tmp_path)
    scan.open_viewer(result)
    (args,) = FakePopen.calls
    assert args[1:3] == ["-m", "cube_solver.sim_view"] and args[3] == str(tmp_path / "scan.json")


def test_no_viewer_without_a_solution(tmp_path, monkeypatch):
    monkeypatch.setattr(scan.subprocess, "Popen", FakePopen)
    FakePopen.calls = []
    result = scan.Result(resolution=None, solution=None, error="parity", wrong=None, folder=tmp_path)
    assert scan.open_viewer(result) is None
    assert FakePopen.calls == []


class FakeTracker:
    """Always locked, on whichever homography the test painted the current frame with."""

    def __init__(self, homographies):
        self.homographies, self.H = iter(homographies), None
        self.det, self.score, self.confidence = None, None, 1.0

    def step(self, frame):
        self.H = next(self.homographies)
        return "tracked"

    def reset(self):
        pass


class FakeCamera:
    def __init__(self, frames):
        self.frames = iter(frames)

    def read(self):
        frame = next(self.frames, None)
        return frame is not None, frame


def test_run_hands_each_finished_scan_to_on_result(tmp_path, monkeypatch):
    monkeypatch.setattr(scan, "SCAN_DIR", tmp_path)
    rng = random.Random(3)
    state = apply_moves(SOLVED, "F2 R' U B")
    colours = dict(zip("URFDLB", (lab_bgr(c) for c in rng.sample(NOMINAL, 6))))
    frames, homographies = [], []
    for face in scan.ORDER:
        k = "URFDLB".index(face)
        H = similarity(rng.uniform(-30, 30))
        frames.append(paint_face(state[9 * k:9 * k + 9], colours, H))
        homographies.append(H @ rng.choice(_SYMMETRIES))
    frames.append(frames[-1])                      # one more frame, on which ESC is pressed
    homographies.append(homographies[-1])
    keys = iter([scan.KEY_SPACE] * 6 + [scan.KEY_ESC])
    monkeypatch.setattr(scan.lt, "LineTracker", lambda: FakeTracker(homographies))
    monkeypatch.setattr(scan.cv2, "imshow", lambda *a: None)
    monkeypatch.setattr(scan.cv2, "waitKey", lambda ms: next(keys))
    got = []
    assert scan.run(FakeCamera(frames), False, None, got.append) == 0
    (result,) = got
    assert result.resolution.state == state
    assert apply_moves(state, result.solution) == SOLVED
    assert set(result.colours) == set("URFDLB")
    assert all(0 <= c <= 1 for rgb in result.colours.values() for c in rgb)


def test_run_returns_1_when_the_camera_stops():
    assert scan.run(FakeCamera([]), False, None, lambda r: None) == 1


def test_main_reports_and_opens_the_viewer(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(scan, "SCAN_DIR", tmp_path)
    rng = random.Random(5)
    state = apply_moves(SOLVED, "U R2 D'")
    colours = dict(zip("URFDLB", (lab_bgr(c) for c in rng.sample(NOMINAL, 6))))
    frames, homographies = [], []
    for face in scan.ORDER:
        k = "URFDLB".index(face)
        H = similarity(rng.uniform(-30, 30))
        frames.append(paint_face(state[9 * k:9 * k + 9], colours, H))
        homographies.append(H)
    frames.append(frames[-1])
    homographies.append(homographies[-1])

    class Camera(FakeCamera):
        released = False

        def isOpened(self):
            return True

        def release(self):
            Camera.released = True

    keys = iter([scan.KEY_SPACE] * 6 + [scan.KEY_ESC])
    monkeypatch.setattr(scan.cv2, "VideoCapture", lambda index: Camera(frames))
    monkeypatch.setattr(scan.lt, "LineTracker", lambda: FakeTracker(homographies))
    monkeypatch.setattr(scan.cv2, "imshow", lambda *a: None)
    monkeypatch.setattr(scan.cv2, "waitKey", lambda ms: next(keys))
    monkeypatch.setattr(scan.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(scan.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(scan.sys, "argv", ["scan.py", "0"])
    FakePopen.calls = []
    assert scan.main() == 0
    assert Camera.released
    assert "solution" in capsys.readouterr().out
    (args,) = FakePopen.calls
    assert args[3].endswith("scan.json")
