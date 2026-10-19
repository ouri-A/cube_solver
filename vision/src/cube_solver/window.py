#!/usr/bin/env python3
"""
The whole scan in one window: the camera on the left, the cube on the right,
the solution along the bottom.

    uv run python -m cube_solver.window              camera 0
    uv run python -m cube_solver.window 1            camera 1
    uv run python -m cube_solver.window --mirror     the camera delivers a mirrored image

While scanning: SPACE read the face, U or BACKSPACE undo, N new scan, R reset the
tracker, D show the tracker's numbers, ESC quit. Each face shows on the cube as it
is read; faces not read yet are grey.
Once solved: SPACE play or pause, RIGHT and LEFT one move, N new scan.
Right-drag turns the cube, scroll zooms.

The camera panel is scan.Scanner's own drawing and the cube is sim_view's; this file
only arranges them and routes keys, so it is the one file that knows about the layout.
Later the left panel shows the robot's camera instead of a hand-held cube.
"""
from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

from cube_solver import scan, sim_view
from cube_solver.cube_model import FACES
from cube_solver.sample import lab_to_bgr

SIZE = (1280, 720)            # px, window
PANEL_H = 0.72                # camera panel height, in UI units (the window is 1 unit tall)
MARGIN = 0.02                 # UI units
BAR_H = 0.15                  # bottom bar height, UI units: the status line and the keys line
UNREAD = (0.35, 0.35, 0.35)   # RGB 0-1, a sticker whose face is not read yet
BACKGROUND = (0.12, 0.13, 0.15)

SCANNING_KEYS = "SPACE read face    U undo    N new scan    R reset tracker    D tracker numbers    ESC quit"
SOLVED_KEYS = "SPACE play/pause    RIGHT / LEFT one move    N new scan    ESC quit    right-drag turns the cube, scroll zooms"


def sticker_colours(session: scan.ScanSession) -> list[tuple[float, float, float]]:
    """RGB 0-1 for all 54 stickers in kociemba order: each read face as measured, the rest grey."""
    out = []
    for f in FACES:
        reading = session.readings.get(f)
        for i in range(9):
            if reading is None:
                out.append(UNREAD)
            else:
                b, g, r = lab_to_bgr(reading.lab_mean[i])
                out.append((r / 255, g / 255, b / 255))
    return out


def status_line(result: scan.Result | None) -> str:
    """What the bottom bar says when no solution is playing."""
    if result is None:
        return "Solution appears after the sixth face."
    return f"Could not solve: {result.error}.  N to scan again."


class FirstLine:
    """Hands Player a text target that keeps only its first line: the keys have their own line here."""

    def __init__(self, text) -> None:
        self._text = text

    @property
    def text(self) -> str:
        return self._text.text

    @text.setter
    def text(self, value: str) -> None:
        self._text.text = value.split("\n")[0]


class CameraPanel:
    """A flat panel in the UI that shows BGR frames. The texture is made on the first frame."""

    def __init__(self, height: float, top_left: tuple[float, float]) -> None:
        from ursina import Entity, camera
        self.height, self.top_left = height, top_left
        self.entity = Entity(parent=camera.ui, model="quad", enabled=False)
        self.texture = None

    def place(self, top_left: tuple[float, float]) -> None:
        """Pin the panel's top-left corner, UI units. Takes effect once the first frame has set its size."""
        self.top_left = top_left
        if self.texture is not None:
            self.entity.position = (top_left[0] + self.entity.scale_x / 2, top_left[1] - self.height / 2)

    def show(self, bgr: np.ndarray) -> None:
        from panda3d.core import Texture as PandaTexture
        from ursina import Texture
        h, w = bgr.shape[:2]
        if self.texture is None or (self.texture.getXSize(), self.texture.getYSize()) != (w, h):
            self.texture = PandaTexture("camera")
            self.texture.setup2dTexture(w, h, PandaTexture.T_unsigned_byte, PandaTexture.F_rgb)
            self.entity.texture = Texture(self.texture)
            width = self.height * w / h
            self.entity.scale = (width, self.height)
            self.place(self.top_left)
            self.entity.enabled = True
        # Panda's rows run bottom-up, OpenCV's top-down; "BGR" says the byte order is OpenCV's
        self.texture.setRamImageAs(np.ascontiguousarray(bgr[::-1]).tobytes(), "BGR")

    @property
    def right(self) -> float:
        """The panel's right edge, UI units."""
        return self.entity.x + self.entity.scale_x / 2


class ScanWindow:
    """Owns the layout. Every frame: one camera frame through the Scanner onto the panel.

    cam is anything with OpenCV's read(); scanner is a scan.Scanner. Both are passed
    in, so a test can drive the window with saved frames.
    """

    def __init__(self, cam, scanner: scan.Scanner) -> None:
        from ursina import Entity, Text, camera, color, window

        self.cam, self.scanner = cam, scanner
        self.exit_code = 0
        self.panel = CameraPanel(PANEL_H, (window.top_left.x + MARGIN, 0.5 - MARGIN))
        self.cube_from = 0.56                       # the 3D view starts at least this far across the window
        self.bar = Entity(parent=camera.ui, model="quad", color=color.rgb(0.07, 0.07, 0.08), z=1)
        self.faces_read = Text("", origin=(0, 0.5), scale=1.1)
        self.status = Text("", scale=1.1)
        self.keys = Text("", scale=0.9, color=color.light_gray)
        self.cubies, self.stickers, self.player = {}, [], None
        self.layout()
        self.new_cube()
        Entity(update=self.update, input=self.input)

    # ------------------------------------------------------------- the cube
    def new_cube(self, state: str | None = None, colours: dict | None = None) -> None:
        """Replace the drawn cube: all grey for a new scan, or a scanned state in its colours."""
        from ursina import destroy
        for cubie in self.cubies.values():
            destroy(cubie)
        rgb = [UNREAD] * 54 if state is None else [colours[letter] for letter in state]
        self.cubies, self.stickers = sim_view.build_cube(rgb)
        self.player = None

    def refresh(self) -> None:
        """Bring the cube and the text in line with the scanner after a key."""
        session, result = self.scanner.session, self.scanner.result
        read = [f for f in scan.ORDER if f in session.readings]
        self.faces_read.text = "Faces read: " + (" ".join(read) if read else "none yet")
        if result is None or not result.solution:
            if self.player is not None:            # 'n' after a solve: start from a grey cube
                self.new_cube()
            sim_view.paint(self.stickers, sticker_colours(session))
            self.status.text = status_line(result)
            self.keys.text = SCANNING_KEYS
        elif self.player is None:
            solution = "" if result.solution == "(already solved)" else result.solution
            self.new_cube(result.resolution.state, result.colours)
            self.player = sim_view.Player(self.cubies, result.resolution.state, solution, FirstLine(self.status))
            self.keys.text = SOLVED_KEYS

    # -------------------------------------------------------- frame and keys
    def update(self) -> None:
        from ursina import application
        self.layout()
        ok, frame = self.cam.read()
        if not ok:
            print("ERROR: camera stopped delivering frames.", file=sys.stderr)
            self.exit_code = 1
            application.quit()
            return
        self.panel.show(self.scanner.step(frame))

    def input(self, key: str) -> None:
        from ursina import application
        if key == "escape":
            application.quit()
        elif self.player is not None and key in ("space", "right arrow", "left arrow"):
            self.player.key(key)
        elif key in ("space", "backspace", "u", "n", "r", "d"):
            result = self.scanner.press(key)
            if result is not None:
                scan.report(result)
            self.refresh()

    def layout(self) -> None:
        """Place everything from the window's current shape, every frame.

        On a resize Ursina moves each UI item's x in proportion but never widens it, so a
        layout fixed at start-up drifts off the edges once the window is maximised.
        """
        from ursina import window
        aspect = window.aspect_ratio
        left = -aspect / 2 + MARGIN
        self.panel.place((left, 0.5 - MARGIN))
        self.bar.scale = (aspect, BAR_H)
        self.bar.position = (0, -0.5 + BAR_H / 2)
        self.status.position = (left, -0.5 + BAR_H - MARGIN)
        self.keys.position = (left, -0.5 + MARGIN + 0.03)
        cube_left = self.fit_cube_view()
        self.faces_read.position = (aspect * ((cube_left + 1) / 2 - 0.5), 0.5 - MARGIN)

    def fit_cube_view(self) -> float:
        """Keep the 3D view to the right of the panel and above the bar, undistorted.

        Returns where the view starts, as a fraction of the window width.
        """
        from ursina import camera, window
        left = max(self.cube_from, (self.panel.right + window.aspect_ratio / 2) / window.aspect_ratio + 0.01) \
            if self.panel.entity.enabled else self.cube_from
        bottom = BAR_H
        camera.display_region.set_dimensions(left, 1.0, bottom, 1.0)
        w, h = window.size
        camera.perspective_lens.set_aspect_ratio((1.0 - left) * w / ((1.0 - bottom) * h))
        return left


def make_window():
    from ursina import Ursina, color, window
    app = Ursina(title="cube_solver", borderless=False, size=SIZE)
    window.color = color.rgb(*BACKGROUND)
    for widget in (window.fps_counter, window.entity_counter, window.collider_counter, window.exit_button,
                   window.cog_button):
        widget.enabled = False                     # Ursina's debug overlay
    from ursina import EditorCamera, camera
    camera.position = (0, 0, -11)
    EditorCamera(rotation=(25, -35, 0))            # looking down on U, F and R
    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="The whole scan in one window.")
    parser.add_argument("source", nargs="?", default="0", help="camera index (default 0)")
    parser.add_argument("--mirror", action="store_true", help="the camera image is mirrored")
    args = parser.parse_args()

    cam = cv2.VideoCapture(int(args.source))
    if not cam.isOpened():
        print(f"ERROR: could not open camera {args.source}.", file=sys.stderr)
        return 1
    app = make_window()
    ui = ScanWindow(cam, scan.Scanner(args.mirror, debug=False))
    ui.refresh()
    try:
        app.run()
    except SystemExit:
        pass
    finally:
        cam.release()
    return ui.exit_code


if __name__ == "__main__":
    sys.exit(main())
