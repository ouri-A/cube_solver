"""
Read the 9 sticker colours of one face from a tracked grid.

The tracker's homography H maps lattice coordinates (0..3, 0..3) to pixels,
but which lattice corner is "top-left" is arbitrary. So first the lattice is
turned (one of the 8 symmetries of a square) until its columns run right and
its rows run down in the image: the face is assumed to be held less than 45
degrees from upright. Then the face is warped to a flat 3x3 image and each
cell is read from a centred patch half a cell wide, which stays inside the
sticker even if the grid is off by a quarter cell and keeps the black borders
out.

Colours are averaged in CIELAB (L 0..100, a and b roughly -127..127), where
brightness sits alone in L.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

CELL_PX = 40                     # warped size of one cell
PATCH = (CELL_PX // 4, 3 * CELL_PX // 4)   # centred half-cell patch, px within a cell

# The 8 symmetries of the square [0,3]^2 as 3x3 matrices on (u, v, 1).
_SYMMETRIES = []
for swap in (False, True):
    for su in (1, -1):
        for sv in (1, -1):
            M = np.array([[su, 0, 0 if su > 0 else 3], [0, sv, 0 if sv > 0 else 3], [0, 0, 1]], float)
            if swap:
                M = M[[1, 0, 2]]
            _SYMMETRIES.append(M)


@dataclass
class FaceReading:
    lab_mean: np.ndarray     # (9, 3) row-major, top-left first, as seen upright
    lab_median: np.ndarray   # (9, 3) same cells, median instead of mean
    spread: np.ndarray       # (9,) mean per-channel standard deviation inside the patch
    flat: np.ndarray         # the warped face, BGR, 3*CELL_PX square


def _to_image(H: np.ndarray, uv) -> np.ndarray:
    p = np.c_[np.asarray(uv, float), np.ones(len(uv))] @ H.T
    return p[:, :2] / p[:, 2:3]


def upright(H: np.ndarray) -> np.ndarray:
    """H composed with the lattice symmetry that makes u point right and v
    point down in the image."""
    best, best_score = None, -np.inf
    for S in _SYMMETRIES:
        G = H @ S
        p = _to_image(G, [(0.5, 1.5), (2.5, 1.5), (1.5, 0.5), (1.5, 2.5)])
        du, dv = p[1] - p[0], p[3] - p[2]
        score = du[0] / np.linalg.norm(du) + dv[1] / np.linalg.norm(dv)
        if score > best_score:
            best, best_score = G, score
    return best


def read_face(frame: np.ndarray, H: np.ndarray, mirror: bool = False) -> FaceReading:
    """frame: BGR image. H: tracker homography, lattice -> pixels.
    mirror: the camera delivers a mirrored image."""
    G = upright(H)
    scale = np.diag([1.0 / CELL_PX, 1.0 / CELL_PX, 1.0])          # flat px -> lattice
    size = 3 * CELL_PX
    flat = cv2.warpPerspective(frame, G @ scale, (size, size),
                               flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    if mirror:
        flat = flat[:, ::-1].copy()
    lab = cv2.cvtColor(flat.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    a, b = PATCH
    means, medians, spreads = [], [], []
    for r in range(3):
        for c in range(3):
            patch = lab[r * CELL_PX + a:r * CELL_PX + b, c * CELL_PX + a:c * CELL_PX + b].reshape(-1, 3)
            means.append(patch.mean(0))
            medians.append(np.median(patch, 0))
            spreads.append(patch.std(0).mean())
    return FaceReading(np.array(means), np.array(medians), np.array(spreads), flat)


def lab_to_bgr(lab) -> tuple[int, int, int]:
    """One LAB colour back to a BGR tuple, for drawing swatches."""
    px = np.asarray(lab, np.float32).reshape(1, 1, 3)
    bgr = cv2.cvtColor(px, cv2.COLOR_LAB2BGR)[0, 0]
    return tuple(int(x) for x in np.clip(bgr * 255, 0, 255))
