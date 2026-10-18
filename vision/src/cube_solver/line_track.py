#!/usr/bin/env python3
"""
Face grid tracked live by snapping it onto the seam lines every frame.

Reuses perception.py's seam stage and track_view.py's detection, fused grid
fit, optical flow and contrast check unchanged, and adds two things:

    1. Line snap  sample 24 points along each of the grid's 8 lines; at each,
                  search sideways (up to 0.25 pitch, shrinking to 0.06) for the
                  strongest seam response; fit a weighted homography to those
                  points; repeat. The response is the greyscale Laplacian
                  after CLAHE, NOT the thresholded mask, so a faint groove
                  that never survives hysteresis still pulls the line.
    2. Detect only when needed (Karpathy's structure). With a grid locked,
                  a frame is: LK flow predicts the grid, the snap corrects it,
                  the line score checks it. Full detection (detect, fused fit,
                  snap the best 3, accept at contrast 0.65) runs only when
                  there is no grid, when the snap fails, and every 10th frame
                  as a recheck, which may jump to a different grid.

Measured 2026-10-03 on the recorded clips, labelled frames scored
(right / wrong / none), device VM timings:

    scrambled, 17 faces   14 / 2 / 1      (track_view: 16 / 1 / 0)
    solved,    21 faces   14 / 1 / 6      (track_view: 11 / 4 / 6)
    empty room, 366       no grid on any frame
    per frame             mean 83 ms, median 40 ms (track_view ~250-290 ms)

    uv run python -m cube_solver.line_track                                   live, camera 0
    uv run python -m cube_solver.line_track 1                                 live, camera 1
    uv run python -m cube_solver.line_track ../data/clips/2026-10-02_122600   play a recorded clip

COLOURS
    green grid    tracked: snapped onto the seams this frame
    cyan grid     acquired, confirmed or jumped by a full detection this frame
    yellow grid   coasting: optical flow only, the snap found no seams

KEYS
    SPACE  pause / resume (clip only); n steps one frame while paused
    r      reset the tracker
    ESC    quit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from cube_solver import perception as pc
from cube_solver import track_view as tv

# ------------------------------------------------------------- parameters
TAU = 20.0             # response = 1 - exp(-Laplacian / TAU): saturates strong edges
SAMPLES = 24           # points per grid line
STEPS = 21             # sideways positions tried per point
ACCEPT = 0.65          # contrast to accept a detected grid (0 wrong of 38 faces)
TOP_N = 3              # fused-fit grids snapped per detection
KEEP_LS = 0.20         # line score that keeps a tracked grid
DECAY = 0.8            # confidence multiplier per coasting frame
DROP = 0.30            # confidence below which the grid is dropped
RECHECK = 10           # frames between full detections while tracking

WINDOW = "line_track"
KEY_ESC = 27
KEY_SPACE = 32
FONT = cv2.FONT_HERSHEY_SIMPLEX
MODE_COLOUR = {"tracked": (0, 220, 0), "acquired": (255, 255, 0), "confirmed": (255, 255, 0),
               "jumped": (255, 255, 0), "coasting": (0, 220, 255)}

_S = (np.arange(SAMPLES) + 0.5) / SAMPLES * 3.0
_LINES = [(np.c_[np.full(SAMPLES, t), _S], (1.0, 0.0)) for t in range(4)] + \
         [(np.c_[_S, np.full(SAMPLES, t)], (0.0, 1.0)) for t in range(4)]
LINE_PTS = np.concatenate([p for p, _ in _LINES])                       # lattice coords
LINE_NRM = np.concatenate([np.tile(n, (SAMPLES, 1)) for _, n in _LINES])  # lattice normals


# ---------------------------------------------------------------- response
def response(frame):
    """(raw, blurred) seam response in 0..1 from the greyscale Laplacian."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    g = pc.CLAHE.apply(g)
    g = cv2.medianBlur(g, pc.MEDIAN_K)
    g = cv2.GaussianBlur(g, (pc.GAUSS_K, pc.GAUSS_K), 0)
    lap = cv2.Laplacian(g, cv2.CV_16S, ksize=pc.LAP_K).astype(np.float32)
    R = (1.0 - np.exp(-np.clip(lap, 0, None) / TAU)).astype(np.float32)
    return R, cv2.GaussianBlur(R, (0, 0), 1.5)


def sample(R, P):
    P = P.astype(np.float32).reshape(1, -1, 2)
    return cv2.remap(R, P[..., 0], P[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0).ravel()


def line_score(H, R):
    """Mean response on the 8 grid lines minus on the 6 half-cell lines."""
    return sample(R, tv.to_image(H, tv.ON_LINES)).mean() - sample(R, tv.to_image(H, tv.OFF_LINES)).mean()


# -------------------------------------------------------------------- snap
def weighted_homography(src, dst, w):
    """Normalised DLT with a weight per point pair."""
    def normalise(P):
        c = P.mean(0)
        s = np.sqrt(2) / max(np.linalg.norm(P - c, axis=1).mean(), 1e-9)
        return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]]), (P - c) * s
    Ts, a = normalise(src)
    Td, b = normalise(dst)
    x, y, u, v = a[:, 0], a[:, 1], b[:, 0], b[:, 1]
    z, o, sw = np.zeros_like(x), np.ones_like(x), np.sqrt(w)[:, None]
    A = np.concatenate([sw * np.c_[-x, -y, -o, z, z, z, u * x, u * y, u],
                        sw * np.c_[z, z, z, -x, -y, -o, v * x, v * y, v]])
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    H = np.linalg.inv(Td) @ Vt[-1].reshape(3, 3) @ Ts
    return H / H[2, 2]


def snap(H, Rb, iterations=3, win_start=0.15, win_end=0.06):
    """Pull grid H onto the seam response. None if there is nothing to pull onto."""
    for it in range(iterations):
        win = win_start + (win_end - win_start) * it / max(iterations - 1, 1)
        d = np.linspace(-win, win, STEPS)
        Q = LINE_PTS[:, None, :] + d[None, :, None] * LINE_NRM[:, None, :]
        img = tv.to_image(H, Q.reshape(-1, 2))
        prof = sample(Rb, img).reshape(len(LINE_PTS), STEPS)
        k = prof.argmax(1)
        rows = np.arange(len(LINE_PTS))
        w = np.clip(prof[rows, k] - np.median(prof, 1), 0, None)
        if (w > 1e-3).sum() < 8:
            return None
        try:
            Hn = weighted_homography(LINE_PTS, img.reshape(len(LINE_PTS), STEPS, 2)[rows, k], w + 1e-6)
        except np.linalg.LinAlgError:
            return None
        if not np.isfinite(Hn).all():
            return None
        H = Hn
    if not tv.PITCH_RANGE[0] <= tv.pitch_of(H) <= tv.PITCH_RANGE[1]:
        return None
    return H


# ----------------------------------------------------------------- tracker
class LineTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.H, self.confidence, self.prev, self.since = None, 0.0, None, 0
        self.det, self.score = None, None

    def detect_grid(self, frame, Rb):
        """Full detection: fused fit, snap its best TOP_N grids, accept at ACCEPT."""
        det = tv.detect(frame)
        self.det = det
        ranked = sorted(tv.candidates(det),
                        key=lambda g: -(g["contrast"] if g["dark"] >= tv.MIN_DARK else -1))[:TOP_N]
        best = None
        for g in ranked:
            H = snap(g["H"], Rb)
            if H is None:
                continue
            contrast, dark = tv.check(H, det)
            if dark >= tv.MIN_DARK and contrast >= ACCEPT and (best is None or contrast > best[1]):
                best = (H, contrast)
        return best

    def step(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        R, Rb = response(frame)
        mode, self.det, self.score = "none", None, None
        if self.H is not None and self.prev is not None:
            M = tv.flow(self.prev, gray, self.H)
            predicted = M @ self.H if M is not None else self.H
            snapped = snap(predicted, Rb, win_start=0.2)
            score = line_score(snapped, R) if snapped is not None else -1.0
            if score >= KEEP_LS:
                self.H, self.confidence, mode, self.score = snapped, 1.0, "tracked", score
            else:
                self.H, self.confidence, mode = predicted, self.confidence * DECAY, "coasting"
                if self.confidence < DROP:
                    self.H, self.confidence, mode = None, 0.0, "dropped"
            self.since += 1
            if self.H is not None and (self.since >= RECHECK or mode == "coasting"):
                self.since = 0
                found = self.detect_grid(frame, Rb)
                if found is not None:
                    far = (np.linalg.norm(tv.centre_of(found[0]) - tv.centre_of(self.H))
                           >= 0.5 * tv.pitch_of(self.H))
                    self.H, self.confidence = found[0], 1.0
                    mode, self.score = ("jumped" if far else "confirmed"), found[1]
        if self.H is None and mode != "dropped":
            found = self.detect_grid(frame, Rb)
            if found is not None:
                self.H, self.confidence, self.since = found[0], 1.0, 0
                mode, self.score = "acquired", found[1]
        self.prev = gray
        return mode


# ------------------------------------------------------------------ drawing
def draw(view, tracker, mode, ms):
    det = tracker.det
    if tracker.H is not None:
        colour = MODE_COLOUR.get(mode, (200, 200, 200))
        pts = np.rint(tv.to_image(tracker.H, tv.ALL16)).astype(int).reshape(4, 4, 2)
        for t in range(4):
            cv2.line(view, tuple(pts[t, 0]), tuple(pts[t, 3]), colour, 2, cv2.LINE_AA)
            cv2.line(view, tuple(pts[0, t]), tuple(pts[3, t]), colour, 2, cv2.LINE_AA)
    lines = [mode + ("" if tracker.score is None else
                     f"  {'line score' if mode == 'tracked' else 'contrast'} {tracker.score:.2f}"),
             f"confidence {tracker.confidence:.2f}",
             f"{ms:.0f} ms" + ("  (detection ran)" if det is not None else "")]
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

    tracker, index, paused = LineTracker(), 0, False
    frame, mode, ms = None, "none", 0.0
    try:
        while True:
            if not paused or frame is None:
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
                mode = tracker.step(frame)
                ms = (time.perf_counter() - started) * 1000.0

            view = frame.copy()
            draw(view, tracker, mode, ms)
            if cam is None:
                cv2.putText(view, f"{index}/{len(frames)}" + ("  PAUSED" if paused else ""),
                            (8, view.shape[0] - 10), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW, view)

            key = cv2.waitKey(1 if not paused else 30) & 0xFF
            if key == KEY_ESC:
                break
            if key == ord("r"):
                tracker.reset()
            elif key == KEY_SPACE and cam is None:
                paused = not paused
            elif key == ord("n") and paused and cam is None:
                frame = None
    finally:
        if cam is not None:
            cam.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
