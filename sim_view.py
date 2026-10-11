#!/usr/bin/env python3
"""
3D view of a cube state.

    uv run sim_view.py scans/<time>/scan.json    the scanned state, in the scan's own colours
    uv run sim_view.py <54-letter state>          any kociemba state, standard colours

Right-drag orbits, scroll zooms.

Frames: cube_model works in x towards R, y towards U, z towards F (right-handed).
Ursina is y-up and left-handed with +z pointing away from the camera, so a
cube_model point (x, y, z) is drawn at (x, y, -z) and F faces the camera.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from cube_model import FACES, STICKERS
from sample import lab_to_bgr

CUBIE = 0.96      # cubie edge, as a fraction of the 1-unit spacing between cubie centres
STICKER = 0.84    # sticker edge, same units
LIFT = 0.505      # sticker centre from its cubie centre, in cubie edges (0.5 = on the surface)

# RGB 0-1, used when there is no scan to take colours from.
STANDARD = {
    "U": (0.95, 0.95, 0.95), "R": (0.80, 0.08, 0.08), "F": (0.05, 0.60, 0.20),
    "D": (1.00, 0.85, 0.00), "L": (1.00, 0.45, 0.00), "B": (0.05, 0.25, 0.80),
}


def to_scene(v) -> tuple[float, float, float]:
    """cube_model frame -> Ursina frame."""
    x, y, z = v
    return (x, y, -z)


def cubie_positions() -> list[tuple[int, int, int]]:
    """The 26 visible cubie centres, in cube_model's frame."""
    return sorted({p for p, _ in STICKERS})


def scan_colours(scan: dict) -> dict[str, tuple[float, float, float]]:
    """Each face letter's colour as RGB 0-1: the mean LAB of every sticker the scan gave that letter."""
    lab = np.concatenate([np.asarray(scan["faces"][f]["lab_mean"], float) for f in FACES])
    letters = np.array(list(scan["state"]))
    out = {}
    for f in FACES:
        b, g, r = lab_to_bgr(lab[letters == f].mean(axis=0))
        out[f] = (r / 255, g / 255, b / 255)
    return out


def load(source: str) -> tuple[str, dict[str, tuple[float, float, float]], str]:
    """A scan.json path or a 54-letter state -> (state, colours, title)."""
    if source.endswith(".json"):
        scan = json.loads(Path(source).read_text())
        return scan["state"], scan_colours(scan), Path(source).parent.name
    state = source.strip().upper()
    if len(state) != 54 or set(state) - set(FACES):
        raise ValueError(f"a state is 54 letters from {FACES}, got {source!r}")
    return state, STANDARD, "state"


def build(state: str, colours: dict) -> dict:
    """Make one black cube per cubie and one coloured quad per sticker. Returns cubie entities by position."""
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


def main() -> int:
    parser = argparse.ArgumentParser(description="3D view of a cube state.")
    parser.add_argument("source", help="scans/<time>/scan.json, or a 54-letter kociemba state")
    args = parser.parse_args()
    try:
        state, colours, title = load(args.source)
    except (OSError, ValueError, KeyError) as e:
        print(f"sim_view: {e}", file=sys.stderr)
        return 1

    app = make_app(f"cube_solver: {title}")
    build(state, colours)
    add_camera_and_title(title)
    app.run()
    return 0


def make_app(window_title: str):
    from ursina import Ursina, color, window

    app = Ursina(title=window_title, borderless=False, size=(900, 900))
    window.color = color.rgb(0.12, 0.13, 0.15)
    for widget in (window.fps_counter, window.entity_counter, window.collider_counter, window.exit_button):
        widget.enabled = False                     # Ursina's debug overlay
    return app


def add_camera_and_title(title: str) -> None:
    from ursina import EditorCamera, Text, camera, window

    camera.position = (0, 0, -11)                  # cube fills about half the window
    EditorCamera(rotation=(25, -35, 0))            # looking down on U, F and R
    Text(title, position=window.top_left + (0.02, -0.02), scale=1.2)


if __name__ == "__main__":
    sys.exit(main())
