#!/usr/bin/env python3
"""
Guided six-face scan: read the cube, resolve its colours, solve it.

Whatever face you show first becomes FRONT, and whatever is on top of it
becomes UP. The screen then walks you through the other five faces:

    F  show any face, its top edge up               SPACE
    R  turn the cube LEFT once                      SPACE
    B  turn it LEFT again                           SPACE
    L  turn it LEFT again                           SPACE
    U  turn LEFT once more (front faces you again),
       then tip the TOP TOWARDS you                 SPACE
    D  tip the top AWAY twice (bottom faces you)    SPACE

A small cube in the instruction bar plays each turn the way it will look
on screen (guide_anim.py): copy it. Hold each face less than 45 degrees from upright. SPACE only reads a face
while the grid is locked (green or cyan). After the sixth face the colours
are resolved (resolve.py), checked, and solved with kociemba; the moves are
for the cube held as it was for the FRONT face.

Every scan is saved to scans/<time>/: scan.json (raw readings, state,
checks, solution) and one PNG per face, so it can be re-resolved offline.

    uv run scan.py                     camera 0
    uv run scan.py 1                   camera 1
    uv run scan.py --mirror            the camera delivers a mirrored image
    uv run scan.py --expect "R U F'"   scramble a solved cube with these moves
                                       (holding it as you will for FRONT),
                                       then scan: wrong stickers are listed

KEYS
    SPACE      read the face
    BACKSPACE  undo the last face (or u)
    r          reset the tracker
    n          new scan (after a result)
    ESC        quit
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import line_track as lt
import guide_anim
from cube_model import FACES, SOLVED, apply_moves
from resolve import Resolution, resolve
from sample import FaceReading, lab_to_bgr, read_face

ORDER = "FRBLUD"
INSTRUCTIONS = {
    "F": ("Show any face: it becomes FRONT.", "Keep its top edge up."),
    "R": ("Turn the cube LEFT once.", "<"),
    "B": ("Turn it LEFT again.", "<"),
    "L": ("Turn it LEFT again.", "<"),
    "U": ("Turn LEFT once more, then", "tip the TOP TOWARDS you."),
    "D": ("Tip the top AWAY twice:", "the bottom faces you."),
}
LOCKED = {"tracked", "acquired", "confirmed", "jumped"}   # tracker modes SPACE accepts
SCAN_DIR = Path("scans")

WINDOW = "scan"
KEY_ESC, KEY_SPACE, KEY_BACKSPACE = 27, 32, 8
FONT = cv2.FONT_HERSHEY_SIMPLEX
SWATCH = 12                      # px per sticker in the progress strip
BANNER = 92                      # px, height of the black instruction bar
GUIDE_PX = 86                    # px, the animated turn guide inside it


@dataclass
class ScanSession:
    """The six readings so far. Pure data: no camera, no window."""
    mirror: bool = False
    readings: dict[str, FaceReading] = field(default_factory=dict)
    frames: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def next_face(self) -> str | None:
        return next((f for f in ORDER if f not in self.readings), None)

    def capture(self, frame: np.ndarray, H: np.ndarray) -> str:
        face = self.next_face
        self.readings[face] = read_face(frame, H, self.mirror)
        self.frames[face] = frame.copy()
        return face

    def undo(self) -> None:
        done = [f for f in ORDER if f in self.readings]
        if done:
            self.readings.pop(done[-1])
            self.frames.pop(done[-1])

    def lab(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """(54, 3) sticker LAB in kociemba order, and centre medians per face."""
        lab = np.concatenate([self.readings[f].lab_mean for f in FACES])
        return lab, {f: self.readings[f].lab_median[4] for f in FACES}


@dataclass
class Result:
    resolution: Resolution
    solution: str | None
    error: str | None
    wrong: list[str] | None      # with --expect: stickers that differ, e.g. "R3 read U want F"
    folder: Path


def finish(session: ScanSession, expect: str | None) -> Result:
    """Resolve, check, solve, compare, save."""
    lab, refs = session.lab()
    res = resolve(lab, refs)
    solution, error = None, None
    if res.check.ok:
        try:
            import kociemba
            solution = kociemba.solve(res.state) or "(already solved)"
        except ValueError as e:
            error = str(e)
    else:
        error = "; ".join(res.check.problems)
    wrong = None
    if expect is not None:
        want = apply_moves(SOLVED, expect)
        wrong = [f"{FACES[i // 9]}{i % 9 + 1} read {g} want {w}"
                 for i, (g, w) in enumerate(zip(res.state, want)) if g != w]
    folder = save(session, res, solution, error, expect, wrong)
    return Result(res, solution, error, wrong, folder)


def save(session, res, solution, error, expect, wrong) -> Path:
    folder = SCAN_DIR / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    data = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "mirror": session.mirror,
        "faces": {f: {"lab_mean": r.lab_mean.round(2).tolist(),
                      "lab_median": r.lab_median.round(2).tolist(),
                      "spread": r.spread.round(2).tolist()}
                  for f, r in session.readings.items()},
        "state": res.state,
        "legal": res.check.ok,
        "problems": res.check.problems,
        "rounds": res.rounds,
        "piece_cost": {k: round(v, 1) for k, v in res.piece_cost.items()},
        "solution": solution,
        "error": error,
        "expect": expect,
        "wrong": wrong,
    }
    (folder / "scan.json").write_text(json.dumps(data, indent=1))
    for f, frame in session.frames.items():
        cv2.imwrite(str(folder / f"{f}.png"), frame)
    return folder


# ----------------------------------------------------------------- drawing
def text(view, s, org, scale=0.6, colour=(255, 255, 255)):
    cv2.putText(view, s, (org[0] + 1, org[1] + 1), FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(view, s, org, FONT, scale, colour, 1, cv2.LINE_AA)


def draw_strip(view, session: ScanSession) -> None:
    """Six small faces top-right in scan order: captured ones as swatches."""
    w = view.shape[1]
    for k, f in enumerate(ORDER):
        x0, y0 = w - (6 - k) * (3 * SWATCH + 8) - 4, 8
        r = session.readings.get(f)
        for i in range(9):
            x, y = x0 + (i % 3) * SWATCH, y0 + (i // 3) * SWATCH
            colour = lab_to_bgr(r.lab_mean[i]) if r is not None else (60, 60, 60)
            cv2.rectangle(view, (x, y), (x + SWATCH - 1, y + SWATCH - 1), colour, -1)
        edge = (0, 255, 255) if f == session.next_face else (0, 0, 0)
        cv2.rectangle(view, (x0 - 1, y0 - 1), (x0 + 3 * SWATCH, y0 + 3 * SWATCH), edge, 1)
        text(view, f, (x0 + SWATCH, y0 + 3 * SWATCH + 14), 0.45)


def draw_prompt(view, session: ScanSession, locked: bool, flash: str | None, t: float) -> None:
    """Instruction bar: the step's text on the left, the animated turn guide
    on the right. t: seconds since this step began, drives the animation."""
    h, w = view.shape[:2]
    face = session.next_face
    cv2.rectangle(view, (0, h - BANNER), (w, h), (0, 0, 0), -1)
    if face is None:
        return
    line1, line2 = INSTRUCTIONS[face]
    text(view, f"{len(session.readings) + 1}/6  {line1}", (10, h - 62))
    text(view, line2 if line2 != "<" else "Copy the cube on the right.", (10, h - 36))
    status = "grid locked: press SPACE" if locked else "no grid yet"
    text(view, status, (10, h - 10), 0.5, (0, 255, 0) if locked else (0, 160, 255))
    known = {f: [lab_to_bgr(c) for c in r.lab_mean] for f, r in session.readings.items()}
    guide = guide_anim.render(face, known, t, session.mirror, GUIDE_PX)
    y0, x0 = h - BANNER + (BANNER - GUIDE_PX) // 2, w - GUIDE_PX - 6
    view[y0:y0 + GUIDE_PX, x0:x0 + GUIDE_PX] = guide
    if flash:
        text(view, flash, (10, h - BANNER - 10), 0.6, (0, 200, 255))


def draw_result(view, result: Result) -> None:
    h, w = view.shape[:2]
    cv2.rectangle(view, (0, h - 120), (w, h), (0, 0, 0), -1)
    if result.solution:
        text(view, "SOLUTION (hold it as for FRONT):", (10, h - 96), 0.55, (0, 255, 0))
        moves = result.solution.split()
        for k in range(0, len(moves), 12):
            text(view, " ".join(moves[k:k + 12]), (10, h - 72 + 22 * (k // 12)), 0.6)
    else:
        text(view, "COULD NOT SOLVE - rescan (n):", (10, h - 96), 0.55, (0, 0, 255))
        text(view, (result.error or "")[:70], (10, h - 72), 0.5)
    if result.wrong is not None:
        text(view, f"vs --expect: {len(result.wrong)} wrong stickers", (10, h - 10), 0.5,
             (0, 255, 0) if not result.wrong else (0, 0, 255))


# --------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description="Guided six-face cube scan.")
    parser.add_argument("source", nargs="?", default="0", help="camera index (default 0)")
    parser.add_argument("--mirror", action="store_true", help="the camera image is mirrored")
    parser.add_argument("--expect", help="moves applied to a solved cube before scanning")
    args = parser.parse_args()

    cam = cv2.VideoCapture(int(args.source))
    if not cam.isOpened():
        print(f"ERROR: could not open camera {args.source}.", file=sys.stderr)
        return 1
    tracker, session = lt.LineTracker(), ScanSession(mirror=args.mirror)
    result, flash, flash_until = None, None, 0.0
    step_face, step_started = None, time.time()      # restart the guide animation on each new step
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                print("ERROR: camera stopped delivering frames.", file=sys.stderr)
                return 1
            started = time.perf_counter()
            mode = tracker.step(frame)
            ms = (time.perf_counter() - started) * 1000.0
            locked = tracker.H is not None and mode in LOCKED

            view = frame.copy()
            if result is None:
                lt.draw(view, tracker, mode, ms)
            draw_strip(view, session)
            if session.next_face != step_face:
                step_face, step_started = session.next_face, time.time()
            if result is None:
                draw_prompt(view, session, locked, flash if time.time() < flash_until else None,
                            time.time() - step_started)
            else:
                draw_result(view, result)
            cv2.imshow(WINDOW, view)

            key = cv2.waitKey(1) & 0xFF
            if key == KEY_ESC:
                break
            if key == ord("r"):
                tracker.reset()
            elif key == ord("n"):
                session, result = ScanSession(mirror=args.mirror), None
            elif key in (KEY_BACKSPACE, ord("u")) and result is None:
                session.undo()
            elif key == KEY_SPACE and result is None:
                if not locked:
                    flash, flash_until = "No grid locked - hold the face still.", time.time() + 1.5
                    continue
                face = session.capture(frame, tracker.H)
                print(f"read {face}")
                if session.next_face is None:
                    result = finish(session, args.expect)
                    report(result)
    finally:
        cam.release()
        cv2.destroyAllWindows()
    return 0


def report(result: Result) -> None:
    res = result.resolution
    print(f"state    {res.state}")
    print("checks   " + ("legal" if res.check.ok else "ILLEGAL: " + "; ".join(res.check.problems)))
    print(f"worst    {res.worst[0]} at {res.worst[1]:.1f} LAB")
    if result.solution:
        print(f"solution {result.solution}")
    if result.error:
        print(f"error    {result.error}")
    if result.wrong is not None:
        print(f"expect   {len(result.wrong)} wrong" + ("" if not result.wrong else ": " + ", ".join(result.wrong)))
    print(f"saved    {result.folder}")


if __name__ == "__main__":
    sys.exit(main())
