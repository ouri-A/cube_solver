#!/usr/bin/env python3
"""
Face grid from crossings + facelet dots, carried between frames by optical flow.

A test program for the 2026-10-02 offline result. It reuses perception.py's
seam, skeleton and crossing stages unchanged and adds three things on top:

    1. Dots      sealed holes in the closed seam mask, kept if 250-9000 px,
                 roughly square (short/long side >= 0.6) and filled
                 (area / min-area rect >= 0.65). A dot sits at a cell centre.
    2. Grid fit  guesses from each dot (its size and angle give one cell, tried
                 at all 9 positions), from crossing pairs and from crossing
                 triples. Each guess is refined against every crossing and dot
                 it lines up with. A grid is a homography from lattice
                 coordinates (face corners at 0..3) to pixels.
                 Checked by CONTRAST: seam white on the 8 grid lines minus
                 white on the 6 half-cell lines between them. A grid is
                 accepted from nothing at 0.60.
    3. Tracking  LK optical flow (rotation + scale + shift) carries the grid
                 to the next frame. If this frame's evidence near it has
                 contrast >= 0.56 it is CONFIRMED; otherwise it COASTS and its
                 confidence decays x0.9 per frame, and it is dropped below 0.3.

Offline on the labelled frames: right grid on 11/21 solved faces and 16/17
scrambled (crossings-only lattice: 2 and 5), no grid on 366 empty-room frames.

    uv run track_view.py                                   live, camera 0
    uv run track_view.py 1                                 live, camera 1
    uv run track_view.py ../data/clips/2026-10-02_122600   play a recorded clip

COLOURS
    green grid    confirmed by this frame's detections
    yellow grid   coasting on optical flow alone
    cyan grid     just acquired, or jumped to a better grid elsewhere
    yellow dots   interior crossings found this frame
    magenta dots  facelet dots found this frame

KEYS
    d      detections (crossings, dots) on / off
    SPACE  pause / resume (clip only); n steps one frame while paused
    r      reset the tracker
    ESC    quit
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from perception import (GAP_MAX, GAP_MIN, MERGE, ARM_R, crossing_number,
                        interior_crossings, merge_points, seam_mask,
                        square_corners, zhang_suen)

# ------------------------------------------------------------- parameters
# Dots
DOT_AREA = (250, 9000)       # px
DOT_SQUARISH = 0.60          # short side / long side of the min-area rect
DOT_FILLED = 0.65            # contour area / min-area rect area
DOT_SIZE = (0.55, 1.25)      # dot side / grid pitch, to count as a match

# Grid fit
MATCH_TOL = 0.22             # pitches; a detection this close to a prediction matches
MIN_SUPPORT = 2              # crossings + dots a guess must explain to be refined
TOP_K = 40                   # guesses refined per frame
PITCH_RANGE = (12, 120)      # px
MIN_DARK = 8                 # of the 9 cell centres that must be black in the mask
ACQUIRE = 0.60               # contrast to accept a grid from nothing
PAIR_ANGLE = math.radians(15)

# Tracking
KEEP = 0.56                  # contrast that confirms the tracked grid
DECAY = 0.9                  # confidence multiplier per unconfirmed frame
DROP = 0.30                  # confidence below which the grid is dropped
NEAR = 0.5                   # pitches; a grid this close to the prediction is the same face
FB_MAX = 1.0                 # px; forward-backward LK error allowed
LK = dict(winSize=(21, 21), maxLevel=3,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

# Drawing
WINDOW = "track_view"
KEY_ESC = 27
KEY_SPACE = 32
FONT = cv2.FONT_HERSHEY_SIMPLEX
MODE_COLOUR = {"confirmed": (0, 220, 0), "coasting": (0, 220, 255),
               "acquired": (255, 220, 0), "jumped": (255, 220, 0)}
CROSS_COLOUR = (0, 255, 255)
DOT_COLOUR = (255, 0, 255)

# ------------------------------------------------------- lattice geometry
INNER = np.array([(i, j) for i in (1, 2) for j in (1, 2)], float)
CELLS = np.array([(a + .5, b + .5) for a in range(3) for b in range(3)], float)
ALL16 = np.array([(i, j) for i in range(4) for j in range(4)], float)
_S = np.linspace(0, 1, 48)
ON_LINES = np.concatenate(
    [np.outer(1 - _S, a) + np.outer(_S, b)
     for a, b in [((t, 0), (t, 3)) for t in range(4)] + [((0, t), (3, t)) for t in range(4)]])
OFF_LINES = np.concatenate(
    [np.outer(1 - _S, a) + np.outer(_S, b)
     for a, b in [((t, 0), (t, 3)) for t in (.5, 1.5, 2.5)] + [((0, t), (3, t)) for t in (.5, 1.5, 2.5)]])
DILATE = np.ones((5, 5), np.uint8)


def to_image(H, ij):
    """Lattice coordinates -> pixels."""
    p = np.c_[ij, np.ones(len(ij))] @ H.T
    return p[:, :2] / p[:, 2:3]


def pitch_of(H):
    c = to_image(H, np.array([(1.5, 1.5), (2.5, 1.5), (1.5, 2.5)]))
    return 0.5 * (np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[0]))


def centre_of(H):
    return to_image(H, np.array([(1.5, 1.5)]))[0]


def affine(o, u, v):
    """Grid with origin o and cell steps u, v."""
    return np.array([[u[0], v[0], o[0]], [u[1], v[1], o[1]], [0, 0, 1]], float)


def angle_gap_90(a, b):
    d = abs(a - b) % (math.pi / 2)
    return min(d, math.pi / 2 - d)


# --------------------------------------------------------------- detection
def find_dots(clean):
    """Sealed holes shaped like a facelet: (x, y, side, angle mod 90)."""
    contours, hierarchy = cv2.findContours(clean, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    dots = []
    if hierarchy is None:
        return np.zeros((0, 4))
    for contour, (_, _, _, parent) in zip(contours, hierarchy[0]):
        if parent < 0:
            continue                                   # outer contour, not a hole
        (cx, cy), (w, h), angle = cv2.minAreaRect(contour)
        area, rect = cv2.contourArea(contour), max(w * h, 1e-6)
        if (DOT_AREA[0] <= area <= DOT_AREA[1]
                and min(w, h) / max(w, h, 1e-6) >= DOT_SQUARISH
                and area / rect >= DOT_FILLED):
            dots.append((cx, cy, math.sqrt(rect), math.radians(angle) % (math.pi / 2)))
    return np.array(dots, float).reshape(-1, 4)


def axis_of(crossing):
    """A crossing's grid direction mod 90 degrees, from its four arms."""
    a = np.array(crossing.arms) * 4.0
    return (math.atan2(np.sin(a).sum(), np.cos(a).sum()) / 4.0) % (math.pi / 2)


def detect(frame):
    clean = seam_mask(frame)
    skel = zhang_suen(clean)
    skel_mask = (skel * 255).astype(np.uint8)
    points = merge_points(crossing_number(skel), MERGE, ARM_R + 2)
    crossings = interior_crossings(skel_mask, clean, points)
    return dict(clean=clean, clean_wide=cv2.dilate(clean, DILATE),
                crossings=crossings,
                cross=np.array([(c.x, c.y) for c in crossings], float).reshape(-1, 2),
                axes=np.array([axis_of(c) for c in crossings]),
                dots=find_dots(clean))


# ------------------------------------------------------------------ guesses
def guesses(det):
    out = []
    for x, y, side, theta in det["dots"]:             # one dot, all 9 cells
        u = np.array([math.cos(theta), math.sin(theta)]) * side
        v = np.array([-u[1], u[0]])
        for a, b in CELLS:
            out.append(affine(np.array([x, y]) - a * u - b * v, u, v))

    X, axes = det["cross"], det["axes"]
    for i in range(len(X)):                            # two crossings
        for j in range(i + 1, len(X)):
            d = X[j] - X[i]
            if not GAP_MIN <= np.linalg.norm(d) <= GAP_MAX * 1.5:
                continue
            if angle_gap_90(axes[i], axes[j]) > PAIR_ANGLE:
                continue
            direction = math.atan2(d[1], d[0]) % (math.pi / 2)
            if angle_gap_90(direction, axes[i]) <= PAIR_ANGLE:          # neighbours
                u, v = d, np.array([-d[1], d[0]])
                out += [affine(X[i] - u - v, u, v), affine(X[i] - u + v, u, -v)]
            elif angle_gap_90(direction + math.pi / 4, axes[i]) <= PAIR_ANGLE:  # diagonal
                c, s = math.cos(-math.pi / 4), math.sin(-math.pi / 4)
                u = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]]) / math.sqrt(2)
                v = np.array([-u[1], u[0]])
                out.append(affine(X[i] - u - v, u, v))

    for p, q, r, _ in square_corners(det["crossings"], 4):   # three crossings
        u, v = X[q] - X[p], X[r] - X[p]
        out.append(affine(X[p] - u - v, u, v))
    return out


# ------------------------------------------------------- refine and check
def match(H, det):
    """Lattice <-> image pairs for every crossing and dot near its prediction."""
    tol = MATCH_TOL * pitch_of(H)
    src, dst = [], []
    if len(det["cross"]):
        pred = to_image(H, INNER)
        d = np.linalg.norm(det["cross"][None] - pred[:, None], axis=2)
        for t, k in enumerate(d.argmin(1)):
            if d[t, k] <= tol:
                src.append(INNER[t]); dst.append(det["cross"][k])
    if len(det["dots"]):
        D, p = det["dots"], pitch_of(H)
        pred = to_image(H, CELLS)
        d = np.linalg.norm(D[None, :, :2] - pred[:, None], axis=2)
        for t, k in enumerate(d.argmin(1)):
            if d[t, k] <= tol and DOT_SIZE[0] * p <= D[k, 2] <= DOT_SIZE[1] * p:
                src.append(CELLS[t]); dst.append(D[k, :2])
    return np.array(src).reshape(-1, 2), np.array(dst).reshape(-1, 2)


def refit(src, dst, H):
    """Homography from 5+ pairs, affine from 3-4, unchanged below that."""
    if len(src) >= 5:
        Hn, _ = cv2.findHomography(src, dst, 0)
        if Hn is not None and np.isfinite(Hn).all() and abs(Hn[2, 2]) > 1e-9:
            return Hn / Hn[2, 2]
    if len(src) >= 3:
        M, *_ = np.linalg.lstsq(np.c_[src, np.ones(len(src))], dst, rcond=None)
        return np.array([[M[0, 0], M[1, 0], M[2, 0]], [M[0, 1], M[1, 1], M[2, 1]], [0, 0, 1]])
    return H


def refine(H, det, iterations=3):
    for _ in range(iterations):
        Hn = refit(*match(H, det), H)
        if not np.isfinite(Hn).all() or np.allclose(Hn, H, atol=1e-3):
            break
        H = Hn
    src, _ = match(H, det)
    return H, len(src)


def white_along(H, mask, lattice_points):
    h, w = mask.shape
    q = np.rint(to_image(H, lattice_points)).astype(int)
    inside = (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
    return mask[q[inside, 1], q[inside, 0]].astype(bool).sum() / len(lattice_points)


def check(H, det):
    """(contrast, dark cells). Contrast = white on the grid lines - white between them."""
    contrast = white_along(H, det["clean_wide"], ON_LINES) - white_along(H, det["clean"], OFF_LINES)
    h, w = det["clean"].shape
    dark = sum(1 for x, y in np.rint(to_image(H, CELLS)).astype(int)
               if 0 <= x < w and 0 <= y < h and det["clean"][y, x] == 0)
    return contrast, dark


def candidates(det, extra=()):
    """Every distinct refined grid this frame supports, checked."""
    scored = []
    for H in guesses(det):
        support = len(match(H, det)[0])
        if support >= MIN_SUPPORT:
            scored.append((support, H))
    scored.sort(key=lambda s: -s[0])
    out = []
    for H in list(extra) + [H for _, H in scored[:TOP_K]]:
        H, _ = refine(H, det)
        if not np.isfinite(H).all():
            continue
        p = pitch_of(H)
        if not PITCH_RANGE[0] <= p <= PITCH_RANGE[1]:
            continue
        c = centre_of(H)
        if any(np.linalg.norm(c - g["centre"]) < 0.3 * p and abs(g["pitch"] / p - 1) < 0.2 for g in out):
            continue
        contrast, dark = check(H, det)
        out.append(dict(H=H, centre=c, pitch=p, contrast=contrast, dark=dark))
    return out


def best_of(grids, threshold):
    ok = [g for g in grids if g["dark"] >= MIN_DARK]
    if not ok:
        return None
    best = max(ok, key=lambda g: g["contrast"])
    return best if best["contrast"] >= threshold else None


# ----------------------------------------------------------------- tracking
def face_region(H, shape, grow=0.4):
    corners = to_image(H, np.array([(-grow, -grow), (3 + grow, -grow),
                                    (3 + grow, 3 + grow), (-grow, 3 + grow)]))
    mask = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(mask, np.rint(corners).astype(np.int32), 255)
    return mask


def flow(prev, cur, H):
    """Similarity transform prev -> cur from LK on corners inside the face."""
    pts = cv2.goodFeaturesToTrack(prev, 80, 0.01, 5, mask=face_region(H, prev.shape))
    if pts is None or len(pts) < 6:
        return None
    nxt, ok1, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, **LK)
    back, ok2, _ = cv2.calcOpticalFlowPyrLK(cur, prev, nxt, None, **LK)
    fb = np.linalg.norm((back - pts).reshape(-1, 2), axis=1)
    good = (ok1.ravel() == 1) & (ok2.ravel() == 1) & (fb < FB_MAX)
    if good.sum() < 6:
        return None
    A, _ = cv2.estimateAffinePartial2D(pts[good].reshape(-1, 2), nxt[good].reshape(-1, 2),
                                       method=cv2.RANSAC, ransacReprojThreshold=2.0)
    return None if A is None else np.vstack([A, [0, 0, 1]])


class Tracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.H, self.confidence, self.prev = None, 0.0, None

    def step(self, frame, det):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mode, contrast = "none", None
        if self.H is not None and self.prev is not None:
            M = flow(self.prev, gray, self.H)
            predicted = M @ self.H if M is not None else self.H
            grids = candidates(det, extra=[predicted])
            p, c = pitch_of(predicted), centre_of(predicted)
            near = [g for g in grids if np.linalg.norm(g["centre"] - c) < NEAR * p and g["dark"] >= MIN_DARK]
            here = max(near, key=lambda g: g["contrast"]) if near else None
            if here is not None and here["contrast"] >= KEEP:
                self.H, self.confidence = here["H"], 1.0
                mode, contrast = "confirmed", here["contrast"]
            else:
                self.H, self.confidence = predicted, self.confidence * DECAY
                mode = "coasting"
                elsewhere = best_of(grids, ACQUIRE)
                if (elsewhere is not None and np.linalg.norm(elsewhere["centre"] - c) >= NEAR * p
                        and (here is None or elsewhere["contrast"] > here["contrast"])):
                    self.H, self.confidence = elsewhere["H"], 1.0
                    mode, contrast = "jumped", elsewhere["contrast"]
                elif self.confidence < DROP:
                    self.H, self.confidence, mode = None, 0.0, "dropped"
        else:
            found = best_of(candidates(det), ACQUIRE)
            if found is not None:
                self.H, self.confidence = found["H"], 1.0
                mode, contrast = "acquired", found["contrast"]
        self.prev = gray
        return mode, contrast


# ------------------------------------------------------------------ drawing
def draw(view, tracker, det, mode, contrast, ms, show_detections):
    if show_detections:
        for x, y in det["cross"]:
            cv2.circle(view, (int(round(x)), int(round(y))), 4, CROSS_COLOUR, -1, cv2.LINE_AA)
        for x, y, _, _ in det["dots"]:
            cv2.circle(view, (int(round(x)), int(round(y))), 3, DOT_COLOUR, -1, cv2.LINE_AA)
    if tracker.H is not None:
        colour = MODE_COLOUR.get(mode, (200, 200, 200))
        pts = np.rint(to_image(tracker.H, ALL16)).astype(int).reshape(4, 4, 2)
        for t in range(4):
            cv2.line(view, tuple(pts[t, 0]), tuple(pts[t, 3]), colour, 2, cv2.LINE_AA)
            cv2.line(view, tuple(pts[0, t]), tuple(pts[3, t]), colour, 2, cv2.LINE_AA)
    lines = [f"{mode}" + (f"  contrast {contrast:.2f}" if contrast is not None else ""),
             f"confidence {tracker.confidence:.2f}",
             f"crossings {len(det['cross'])}  dots {len(det['dots'])}",
             f"{ms:.0f} ms"]
    for k, text in enumerate(lines):
        y = 20 + 20 * k
        cv2.putText(view, text, (9, y + 1), FONT, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(view, text, (8, y), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


# --------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1].strip())
    parser.add_argument("source", nargs="?", default="0",
                        help="camera index, or a folder of frame PNGs (default 0)")
    args = parser.parse_args()

    cam, frames = None, []
    if args.source.isdigit():
        cam = cv2.VideoCapture(int(args.source))
        if not cam.isOpened():
            print(f"ERROR: could not open camera {args.source}.", file=sys.stderr)
            return 1
    else:
        folder = Path(args.source).expanduser().resolve()
        frames = sorted(folder.glob("frame_*.png")) or sorted(folder.glob("*.png"))
        if not frames:
            print(f"ERROR: no PNG frames in {folder}.", file=sys.stderr)
            return 1
        print(f"{len(frames)} frames from {folder}")

    tracker, index, paused, show_detections = Tracker(), 0, False, True
    frame = None
    try:
        while True:
            advance = not paused
            if advance or frame is None:
                if cam is not None:
                    ok, frame = cam.read()
                    if not ok:
                        print("ERROR: camera stopped delivering frames.", file=sys.stderr)
                        return 1
                else:
                    if index >= len(frames):
                        break
                    frame = cv2.imread(str(frames[index]))
                    index += 1
                started = time.perf_counter()
                det = detect(frame)
                mode, contrast = tracker.step(frame, det)
                ms = (time.perf_counter() - started) * 1000.0

            view = frame.copy()
            draw(view, tracker, det, mode, contrast, ms, show_detections)
            if cam is None:
                cv2.putText(view, f"{index}/{len(frames)}" + ("  PAUSED" if paused else ""),
                            (8, view.shape[0] - 10), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW, view)

            key = cv2.waitKey(1 if not paused else 30) & 0xFF
            if key == KEY_ESC:
                break
            if key == ord("d"):
                show_detections = not show_detections
            elif key == ord("r"):
                tracker.reset()
            elif key == KEY_SPACE and cam is None:
                paused = not paused
            elif key == ord("n") and paused and cam is None:
                frame = None                      # forces one new frame next loop
    finally:
        if cam is not None:
            cam.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
