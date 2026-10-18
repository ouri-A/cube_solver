#!/usr/bin/env python3
"""
3D view of a cube state, playing its solution.

    uv run sim_view.py scans/<time>/scan.json    the scanned state in the scan's own colours, its saved solution
    uv run sim_view.py <54-letter state>          any kociemba state, standard colours, solved by kociemba

    space   play / pause        right   one move forward        left   one move back
    Right-drag orbits, scroll zooms.

Frames: cube_model works in x towards R, y towards U, z towards F (right-handed).
Ursina is y-up and left-handed with +z pointing away from the camera, so a
cube_model point (x, y, z) is drawn at (x, y, -z) and F faces the camera.

Which cubies a turn moves, and where they end up, comes from Pieces below, which
is plain numpy and tested against cube_model.apply_moves. Ursina only animates
between those poses.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from colours import face_colours
from cube_model import FACES, SOLVED, STICKERS, apply_moves

CUBIE = 0.96      # cubie edge, as a fraction of the 1-unit spacing between cubie centres
STICKER = 0.84    # sticker edge, same units
LIFT = 0.505      # sticker centre from its cubie centre, in cubie edges (0.5 = on the surface)
TURN_TIME = 0.35  # seconds per quarter turn; a half turn takes twice as long
GAP_TIME = 0.15   # seconds between moves while playing

# RGB 0-1, used when there is no scan to take colours from.
STANDARD = {
    "U": (0.95, 0.95, 0.95), "R": (0.80, 0.08, 0.08), "F": (0.05, 0.60, 0.20),
    "D": (1.00, 0.85, 0.00), "L": (1.00, 0.45, 0.00), "B": (0.05, 0.25, 0.80),
}
NORMAL = {f: np.array(STICKERS[9 * k + 4][1]) for k, f in enumerate(FACES)}   # outward normal of each face
QUARTERS = {"": 1, "2": 2, "'": 3}                                             # clockwise quarter turns


def to_scene(v) -> tuple[float, float, float]:
    """cube_model frame -> Ursina frame."""
    x, y, z = v
    return (x, y, -z)


def cubie_positions() -> list[tuple[int, int, int]]:
    """The 26 visible cubie centres, in cube_model's frame."""
    return sorted({p for p, _ in STICKERS})


def inverse(moves: str) -> str:
    """The moves that undo `moves`."""
    flip = {"": "'", "'": "", "2": "2"}
    return " ".join(m[0] + flip[m[1:]] for m in reversed(moves.split()))


# ---------------------------------------------------------------- the pieces
def clockwise(face: str, quarters: int) -> np.ndarray:
    """Integer rotation matrix: `quarters` clockwise quarter turns of `face`, seen from outside it."""
    a = NORMAL[face]
    cross = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    quarter = np.outer(a, a) - cross                   # v -> a(a.v) - a x v, i.e. -90 degrees about a
    return np.linalg.matrix_power(quarter, quarters)


class Pieces:
    """Where every cubie is and how it is turned, keyed by its solved position."""

    def __init__(self) -> None:
        self.pos = {h: np.array(h) for h in cubie_positions()}
        self.rot = {h: np.eye(3, dtype=int) for h in cubie_positions()}

    def turn(self, move: str) -> list[tuple[int, int, int]]:
        """Apply one move, e.g. "R'". Returns the cubies it moved."""
        face, quarters = move[0], QUARTERS[move[1:]]
        moved = [h for h, p in self.pos.items() if p @ NORMAL[face] == 1]
        R = clockwise(face, quarters)
        for h in moved:
            self.pos[h] = R @ self.pos[h]
            self.rot[h] = R @ self.rot[h]
        return moved

    def state(self, start: str) -> str:
        """The 54-letter state these poses make of a cube that began as `start`."""
        where = {s: k for k, s in enumerate(STICKERS)}
        out = [""] * 54
        for k, (p, n) in enumerate(STICKERS):
            R = self.rot[p]
            out[where[(tuple(self.pos[p]), tuple(R @ np.array(n)))]] = start[k]
        return "".join(out)


# ------------------------------------------------------------------ loading
def scan_colours(scan: dict) -> dict[str, tuple[float, float, float]]:
    """Each face letter's colour as RGB 0-1, from a loaded scan.json (see colours.face_colours)."""
    return face_colours({f: scan["faces"][f]["lab_mean"] for f in FACES}, scan["state"])


def load(source: str) -> tuple[str, dict[str, tuple[float, float, float]], str, str]:
    """A scan.json path or a 54-letter state -> (state, colours, title, solution)."""
    if source.endswith(".json"):
        scan = json.loads(Path(source).read_text())
        if not scan.get("solution"):
            raise ValueError(f"{source} has no solution: {scan.get('error')}")
        solution = "" if scan["solution"] == "(already solved)" else scan["solution"]
        return scan["state"], scan_colours(scan), Path(source).parent.name, solution
    state = source.strip().upper()
    if len(state) != 54 or set(state) - set(FACES):
        raise ValueError(f"a state is 54 letters from {FACES}, got {source!r}")
    import kociemba
    solution = "" if state == SOLVED else kociemba.solve(state)
    return state, STANDARD, "state", solution


# --------------------------------------------------------------------- view
def build(state: str, colours: dict) -> dict:
    """Make one black cube per cubie and one coloured quad per sticker. Returns cubie entities by solved position."""
    from ursina import Entity, color

    cubies = {p: Entity(model="cube", color=color.black, position=to_scene(p), scale=CUBIE)
              for p in cubie_positions()}
    for k, (p, n) in enumerate(STICKERS):
        axis = int(np.flatnonzero(n)[0])
        Entity(parent=cubies[p], model="quad", double_sided=True,
               color=color.rgb(*colours[state[k]]),
               position=tuple(LIFT * c for c in to_scene(n)),
               rotation=[(0, 90, 0), (90, 0, 0), (0, 0, 0)][axis],   # quad is in xy; turn it to face its normal
               scale=STICKER / CUBIE)
    return cubies


class Player:
    """Steps through a solution, animating each turn of the drawn cubies."""

    def __init__(self, cubies: dict, start: str, solution: str, hud) -> None:
        self.cubies, self.start, self.moves, self.hud = cubies, start, solution.split(), hud
        self.pieces = Pieces()
        self.done = 0            # moves applied so far
        self.busy = False        # a turn is animating
        self.playing = False
        self.show()

    def show(self) -> None:
        line = " ".join(f"[{m}]" if i == self.done else m for i, m in enumerate(self.moves))
        if self.done == len(self.moves):
            solved = apply_moves(self.start, " ".join(self.moves)) == SOLVED
            line += "   solved" if solved else "   NOT SOLVED"
        self.hud.text = f"move {self.done}/{len(self.moves)}   {line}\nspace play/pause   right step   left back"

    def step(self, forward: bool = True) -> None:
        if self.busy or not (0 <= self.done + (0 if forward else -1) < len(self.moves)):
            self.playing = False
            return
        move = self.moves[self.done] if forward else inverse(self.moves[self.done - 1])
        self.done += 1 if forward else -1
        self.animate(move)

    def animate(self, move: str) -> None:
        from ursina import Entity, curve, destroy, invoke, scene

        moved = self.pieces.turn(move)
        quarters = {"": 1, "2": 2, "'": -1}[move[1:]]
        n = NORMAL[move[0]]
        axis = int(np.flatnonzero(n)[0])
        # +90 on Ursina's rotation_x, _y or _z is a clockwise quarter turn seen from cube_model's
        # +x, +y or +z side (measured offline), so the sign is the normal's sign in cube_model's frame.
        angle = 90 * quarters * n[axis]
        duration = TURN_TIME * abs(quarters)
        pivot = Entity()
        for h in moved:
            self.cubies[h].world_parent = pivot
        pivot.animate("rotation_" + "xyz"[axis], angle, duration=duration, curve=curve.in_out_sine)

        def finish() -> None:
            setattr(pivot, "rotation_" + "xyz"[axis], angle)   # land exactly, even if a slow frame left
            for h in moved:                                       # the animation short of its end
                c = self.cubies[h]
                c.world_parent = scene
                c.position = to_scene(self.pieces.pos[h])   # snap off float drift; the rotation keeps
                                                            # its float error (~1e-7 rad a turn), harmless
            destroy(pivot)
            self.busy = False
            self.show()
            if self.playing:
                invoke(self.step, delay=GAP_TIME)

        self.busy = True
        invoke(finish, delay=duration + 0.02)

    def key(self, key: str) -> None:
        if key == "space":
            self.playing = not self.playing
            if self.playing:
                self.step()
        elif key in ("right arrow", "left arrow"):
            self.playing = False
            self.step(forward=key == "right arrow")


def main() -> int:
    parser = argparse.ArgumentParser(description="3D view of a cube state, playing its solution.")
    parser.add_argument("source", help="scans/<time>/scan.json, or a 54-letter kociemba state")
    args = parser.parse_args()
    try:
        state, colours, title, solution = load(args.source)
    except (OSError, ValueError, KeyError) as e:
        print(f"sim_view: {e}", file=sys.stderr)
        return 1

    from ursina import Entity

    app = make_app(f"cube_solver: {title}")
    cubies = build(state, colours)
    hud = add_camera_and_title(title)
    player = Player(cubies, state, solution, hud)
    Entity(input=player.key)                       # Ursina calls an entity's input() on every key press
    app.run()
    return 0


def make_app(window_title: str):
    from ursina import Ursina, color, window

    app = Ursina(title=window_title, borderless=False, size=(900, 900))
    window.color = color.rgb(0.12, 0.13, 0.15)
    for widget in (window.fps_counter, window.entity_counter, window.collider_counter, window.exit_button):
        widget.enabled = False                     # Ursina's debug overlay
    return app


def add_camera_and_title(title: str):
    """Camera, title, and the text line the player writes to. Returns that text entity."""
    from ursina import EditorCamera, Text, camera, window

    camera.position = (0, 0, -11)                  # cube fills about half the window
    EditorCamera(rotation=(25, -35, 0))            # looking down on U, F and R
    Text(title, position=window.top_left + (0.02, -0.02), scale=1.2)
    return Text("", position=window.bottom_left + (0.02, 0.08), scale=0.9)


if __name__ == "__main__":
    sys.exit(main())
