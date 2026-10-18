"""
Animated turn guide: a small 3D cube that shows the next turn as it will
look ON SCREEN, so nobody has to work out whose left is meant.

Faces already scanned are drawn in their real colours, so the person can
match the drawing to the cube in their hand; faces not yet scanned are grey.
The face to be read next is outlined in yellow. The cube is drawn from a
three-quarter view (a little from above and from the right) so the face
coming round is visible before it arrives.

Poses are rotations of the cube in the camera frame (x right, y up, z towards
the camera); cube coordinates are cube_model's (x towards R, y towards U,
z towards F). Turning the cube left brings R to the camera: -90 deg about y.
Tipping the top towards you brings U to the camera: +90 deg about x.
"""
from __future__ import annotations

import cv2
import numpy as np

from cube_solver.cube_model import _AXES

SIZE = 220                       # px, square inset
HOLD_START, MOVE, HOLD_END = 0.4, 1.8, 0.8   # s per loop of one step's animation
VIEW_YAW, VIEW_TILT = -28.0, 22.0             # deg: three-quarter view
BODY, UNKNOWN, NEXT = (25, 25, 25), (95, 95, 95), (0, 255, 255)


def rot_x(deg: float) -> np.ndarray:
    t = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])


def rot_y(deg: float) -> np.ndarray:
    t = np.radians(deg)
    return np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])


# Each step: the moves that take the cube from the previous face to this one,
# as (axis, degrees) in the camera frame, applied one after the other.
MOVES = {
    "F": [],
    "R": [("y", -90)],
    "B": [("y", -90)],
    "L": [("y", -90)],
    "U": [("y", -90), ("x", 90)],
    "D": [("x", -180)],
}
ORDER = "FRBLUD"


def _rot(axis: str, deg: float) -> np.ndarray:
    return rot_x(deg) if axis == "x" else rot_y(deg)


def pose_before(face: str) -> np.ndarray:
    """Cube orientation when the step that reads `face` begins."""
    M = np.eye(3)
    for f in ORDER[:ORDER.index(face)]:
        for axis, deg in MOVES[f]:
            M = _rot(axis, deg) @ M
    return M


def pose_at(face: str, t: float) -> np.ndarray:
    """Orientation at time t (s) into the looping animation for `face`."""
    M = pose_before(face)
    moves = MOVES[face]
    if not moves:
        return M
    period = HOLD_START + MOVE * len(moves) + HOLD_END
    t = t % period - HOLD_START
    for axis, deg in moves:
        k = float(np.clip(t / MOVE, 0.0, 1.0))
        k = k * k * (3 - 2 * k)                  # ease in and out
        M = _rot(axis, deg * k) @ M
        t -= MOVE
    return M


def _sticker_quads(face: str):
    """The 9 sticker outlines of a face in cube coordinates, row-major as in
    cube_model, plus the face outline."""
    n, right, down = (np.array(v, float) for v in _AXES[face])
    quads = []
    for r in range(3):
        for c in range(3):
            x0, y0 = -1 + 2 * c / 3, -1 + 2 * r / 3
            pad = 0.06
            corners = [(x0 + pad, y0 + pad), (x0 + 2 / 3 - pad, y0 + pad),
                       (x0 + 2 / 3 - pad, y0 + 2 / 3 - pad), (x0 + pad, y0 + 2 / 3 - pad)]
            quads.append(np.array([n + right * x + down * y for x, y in corners]))
    outline = np.array([n + right * x + down * y for x, y in [(-1, -1), (1, -1), (1, 1), (-1, 1)]])
    return quads, outline


_QUADS = {f: _sticker_quads(f) for f in "URFDLB"}


def render(face: str, colours: dict[str, list[tuple[int, int, int]]], t: float,
           mirror: bool = False, size: int = SIZE) -> np.ndarray:
    """BGR image of the cube at time t of the animation for the step reading
    `face`. colours: face letter -> 9 BGR tuples for faces already scanned."""
    img = np.zeros((size, size, 3), np.uint8)
    view = rot_x(VIEW_TILT) @ rot_y(VIEW_YAW)
    M = view @ pose_at(face, t)
    scale = size * 0.27

    def project(P):
        P = P @ M.T
        return np.c_[size / 2 + scale * P[:, 0], size / 2 - scale * P[:, 1]]

    for f, (quads, outline) in _QUADS.items():
        normal = M @ np.array(_AXES[f][0], float)
        if normal[2] <= 1e-6:                    # facing away from the viewer
            continue
        cv2.fillConvexPoly(img, np.rint(project(outline)).astype(np.int32), BODY, cv2.LINE_AA)
        for k, q in enumerate(quads):
            colour = colours[f][k] if f in colours else UNKNOWN
            cv2.fillConvexPoly(img, np.rint(project(q)).astype(np.int32), colour, cv2.LINE_AA)
        if f == face:
            cv2.polylines(img, [np.rint(project(outline)).astype(np.int32)], True, NEXT,
                          3 if size >= 150 else 2, cv2.LINE_AA)
    if mirror:
        img = img[:, ::-1].copy()
    return img
