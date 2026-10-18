#!/usr/bin/env python3
"""
Cube face perception: find the 4x4 lattice of one cube face in a webcam frame.

Extracted from debug_tees.py on 2026-10-02, keeping only the path that
produced the result. That program and the three it imports from are left
untouched as the record of how each stage was chosen; the measurements behind
every number here are in PROJECT_STATE.md.

PIPELINE

    1. Seams     grey -> CLAHE clip 4, 8x8 tiles -> median 3 -> gaussian 3
                 -> Laplacian k3 (positive) -> hysteresis 12/68
                 -> closing 3x3 ellipse x3. Without CLAHE the hysteresis is
                 5/30, the pipeline before 2026-10-02.
    2. Junctions Zhang-Suen thinning, crossing number at every skeleton
                 pixel, CN >= 3 merged within MERGE px. A thinner splits every
                 X into two adjacent Ts, so CN == 4 never fires on its own.
    3. Crossings four arms on the skeleton ring, all four gaps within PERP of
                 90 degrees, and a facelet (a hole in the closed mask) in each
                 of the four gaps.
    4. Squares   three crossings at distances s, s, sqrt2*s. Distances alone
                 force the right angle, and both tolerances are ratios, so the
                 test is scale-free. Every point of a valid triple is kept.
    5. Lattice   a triple q-p-r gives the fourth vertex at q + r - p. Extending
                 the square's four sides one step lands on the eight edge
                 midpoints (T junctions); extending its two diagonals lands on
                 the four face corners (two arms at a right angle). The best
                 square is the one whose six lines find the most.

KEYS
    g      square and its six lines on / off
    SPACE  save the camera frame and the drawn view
    i      freeze and annotate freehand (e erase, u undo, c clear,
           SPACE save both, ESC discard)
    ESC    quit
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# The package sits one level below the project root; captures/ belongs to the
# project, not to the package.
PROJECT_ROOT = Path(__file__).resolve().parents[3]   # src/cube_solver/ -> vision/ -> repo root
DEFAULT_CAPTURE_DIR = PROJECT_ROOT / "captures"

WINDOW = "perception"
CONTROLS = "controls"
KEY_ESC = 27
KEY_SPACE = 32

# ---------------------------------------------------------------- parameters
# Stage 1, seams.
MEDIAN_K = 3
GAUSS_K = 3
LAP_K = 3
HYST_LO = 5
HYST_HI = 30
CLOSE_K = 3
CLOSE_ITERATIONS = 3          # 2 -> 3 on 2026-10-02: interior crossings surviving 51 -> 65 of 160
# CLAHE before everything else. Measured 2026-10-02 on 40 labelled faces:
# interior crossings with all four seams intact 41 -> 60 of 160 at the same
# clutter (16% white outside the face), confirmed on held-out halves. The
# hysteresis pair is the measured 8/48 put back on the Laplacian's own scale.
CLAHE_CLIP = 4.0
CLAHE_TILES = 8
HYST_LO_CLAHE = 12
HYST_HI_CLAHE = 68

# Stage 2, junctions.
MAX_THIN_PASSES = 60       # only bounds a pathological frame
MERGE = 4                  # px; CN >= 3 pixels closer than this are one point

# Stage 3, crossings.
ARM_R = 6                  # px; ring on the skeleton that reads arm directions
RING_SAMPLES = 32          # one sample every ~11 degrees
PERP = 34                  # degrees any gap between arms may be off 90
QUAD = 160                 # facelet probe distance, % of ARM_R

# Stage 4, squares.
GAP_MIN = 18               # px; closer than this two points are one crossing
GAP_MAX = 120              # px; a compute bound, not a claim about cubes
HYP_TOL = 2                # % the hypotenuse may differ from sqrt2 * leg

# Stage 5, lattice.
MAX_SQUARES = 60           # squares tried per frame

# Live sliders: the two numbers still being tuned on camera.
TRACKBARS = [
    # name       default  maximum
    ("side_tol",      2,      60),   # % the two legs may differ by
    ("snap",          6,      40),   # % of a leg; search radius around a lattice point
]

# ------------------------------------------------------------------- drawing
CROSS_COLOUR = (0, 255, 255)     # interior crossing, detected
FOURTH_COLOUR = (60, 170, 255)   # computed fourth vertex, never detected
TEE_COLOUR = (80, 200, 80)       # edge midpoint
CORNER_COLOUR = (255, 120, 200)  # face corner
LINE_COLOUR = (90, 90, 90)

INK = (255, 0, 255)              # annotation; no cube face is magenta
INK_WIDTH = 2
ERASE_RADIUS = 12

FONT = cv2.FONT_HERSHEY_SIMPLEX
HUD_SCALE = 0.5
HUD_STEP = 20


# ======================================================== stage 1: seams
def seam_mask(frame, use_clahe=True):
    """Camera frame -> closed binary mask of the cube's seams, 0/255 uint8.

    The Laplacian is kept signed-positive and clipped, not normalised, so the
    hysteresis thresholds mean the same thing in every frame. At ksize 3 the
    kernel gain is 1, so no rescaling is needed.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if use_clahe:
        gray = CLAHE.apply(gray)
        lo, hi = HYST_LO_CLAHE, HYST_HI_CLAHE
    else:
        lo, hi = HYST_LO, HYST_HI
    gray = cv2.medianBlur(gray, MEDIAN_K)
    gray = cv2.GaussianBlur(gray, (GAUSS_K, GAUSS_K), 0)

    lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=LAP_K, scale=1.0)
    mag = np.clip(lap, 0, 255).astype(np.uint8)

    edges = hysteresis(mag, lo, hi)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_K, CLOSE_K))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel,
                            iterations=CLOSE_ITERATIONS)


CLAHE = cv2.createCLAHE(clipLimit=CLAHE_CLIP,
                        tileGridSize=(CLAHE_TILES, CLAHE_TILES))


def hysteresis(mag, lo, hi):
    """Keep weak pixels only where their component also holds a strong one."""
    weak = (mag >= lo).astype(np.uint8)
    strong = mag >= hi
    count, labels = cv2.connectedComponents(weak, connectivity=8)
    if count <= 1:
        return np.zeros_like(weak)
    keep = np.zeros(count, dtype=bool)
    keep[labels[strong]] = True
    keep[0] = False
    return np.where(keep[labels], 255, 0).astype(np.uint8)


# ===================================================== stage 2: junctions
# P2..P9, clockwise from north, as (dy, dx). Zhang-Suen's numbering.
NEIGHBOURS = [(-1, 0), (-1, 1), (0, 1), (1, 1),
              (1, 0), (1, -1), (0, -1), (-1, -1)]


def neighbour_codes(img):
    """Each pixel's eight neighbours packed into one byte: bit i is P(i+2).

    One filter2D does it for the whole frame. The border wraps, which is what
    the earlier np.roll version did, so the skeleton is bit-identical to it.
    """
    padded = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_WRAP)
    codes = cv2.filter2D(padded, -1, CODE_KERNEL, borderType=cv2.BORDER_CONSTANT)
    return codes[1:-1, 1:-1]


def _build_tables():
    """Every decision below depends only on a pixel's eight neighbours, so it
    is worked out once for all 256 patterns and looked up per pixel after."""
    kernel = np.zeros((3, 3), np.float32)
    for i, (dy, dx) in enumerate(NEIGHBOURS):
        kernel[1 + dy, 1 + dx] = 1 << i

    delete = [np.zeros(256, np.uint8), np.zeros(256, np.uint8)]
    cn = np.zeros(256, np.uint8)
    for code in range(256):
        p = [(code >> i) & 1 for i in range(8)]
        ring = p + [p[0]]
        b = sum(p)                                           # white neighbours
        a = sum(ring[i] == 0 and ring[i + 1] == 1 for i in range(8))
        cn[code] = sum(abs(ring[i] - ring[i + 1]) for i in range(8)) // 2
        corner = [(p[0] * p[2] * p[4] == 0 and p[2] * p[4] * p[6] == 0),   # N*E*S, E*S*W
                  (p[0] * p[2] * p[6] == 0 and p[0] * p[4] * p[6] == 0)]   # N*E*W, N*S*W
        for step in (0, 1):
            delete[step][code] = 2 <= b <= 6 and a == 1 and corner[step]
    return kernel, delete, cn


CODE_KERNEL, DELETE_TABLES, CN_TABLE = _build_tables()


def zhang_suen(mask):
    """Thin to one-pixel strokes. Returns a 0/1 uint8 skeleton.

    A pixel is deleted when it has 2-6 white neighbours, exactly one 0->1
    transition round its ring (so deleting it cannot split the stroke), and
    the two corner conditions of the current sub-pass. The sub-passes peel
    opposite sides so the skeleton stays centred.
    """
    img = (mask > 0).astype(np.uint8)
    for _ in range(MAX_THIN_PASSES):
        removed = 0
        for table in DELETE_TABLES:
            kill = cv2.LUT(neighbour_codes(img), table) & img
            hits = cv2.countNonZero(kill)
            if hits:
                img -= kill
                removed += hits
        if removed == 0:
            break
    return img


def crossing_number(skel):
    """CN at every skeleton pixel: 1 ending, 2 line, 3 T, 4 crossing."""
    cn = cv2.LUT(neighbour_codes(skel), CN_TABLE)
    return np.where(skel == 1, cn, 0).astype(np.int16)


def merge_points(cn_map, radius, margin):
    """CN >= 3 pixels merged into one point per cluster, as (x, y) floats.

    Dilating by the radius and labelling makes each cluster one component.
    The centroid is taken over the original pixels, not the dilated blob.
    Pixels within `margin` of the border are dropped, because np.roll wraps.
    """
    hits = (cn_map >= 3).astype(np.uint8)
    hits[:margin, :] = hits[-margin:, :] = 0
    hits[:, :margin] = hits[:, -margin:] = 0
    if not hits.any():
        return []

    k = max(int(radius), 1) * 2 + 1
    grown = cv2.dilate(hits, np.ones((k, k), np.uint8))
    count, labels = cv2.connectedComponents(grown, 8)

    ys, xs = np.nonzero(hits)
    ids = labels[ys, xs]
    points = []
    for group in range(1, count):
        sel = ids == group
        if sel.any():
            points.append((float(xs[sel].mean()), float(ys[sel].mean())))
    return points


# ===================================================== stage 3: crossings
@dataclass
class Cross:
    x: float
    y: float
    arms: list = field(default_factory=list)   # arm directions, radians


def ring_offsets(radius, samples=RING_SAMPLES):
    """Integer (dy, dx) steps around a circle, consecutive duplicates removed."""
    out = []
    for i in range(samples):
        angle = 2.0 * math.pi * i / samples
        step = (int(round(radius * math.sin(angle))),
                int(round(radius * math.cos(angle))))
        if not out or step != out[-1]:
            out.append(step)
    if len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return out


def ring_arms(mask, x, y, radius):
    """Direction of each run of white around a ring at (x, y), in radians.

    Each run's direction is the mean of its samples as unit vectors, which
    survives the wrap from 2pi back to 0.
    """
    offsets = ring_offsets(radius)
    height, width = mask.shape
    bits, angles = [], []
    for dy, dx in offsets:
        bits.append(mask[min(max(y + dy, 0), height - 1),
                         min(max(x + dx, 0), width - 1)] > 0)
        angles.append(math.atan2(dy, dx))
    count = len(bits)
    if all(bits) or not any(bits):
        return []

    start = bits.index(False)
    bits = bits[start:] + bits[:start]
    angles = angles[start:] + angles[:start]

    arms = []
    i = 0
    while i < count:
        if bits[i]:
            j = i
            while j < count and bits[j]:
                j += 1
            sx = sum(math.cos(a) for a in angles[i:j])
            sy = sum(math.sin(a) for a in angles[i:j])
            arms.append(math.atan2(sy, sx))
            i = j
        else:
            i += 1
    return arms


def gaps_between(arms):
    """Degrees between consecutive arm directions, all the way round."""
    ordered = sorted(a % (2.0 * math.pi) for a in arms)
    ordered.append(ordered[0] + 2.0 * math.pi)
    return [math.degrees(b - a) for a, b in zip(ordered, ordered[1:])]


def quadrants_dark(mask, x, y, arms, distance) -> bool:
    """Is there a facelet in each of the four gaps between the arms?

    Step out along each bisector; a facelet is a hole in the closed mask, so
    the sample must be black. The only test that knows it is looking at a
    cube rather than at any four-way junction.
    """
    height, width = mask.shape
    ordered = sorted(a % (2.0 * math.pi) for a in arms)
    for a, b in zip(ordered, ordered[1:] + [ordered[0] + 2.0 * math.pi]):
        mid = (a + b) / 2.0
        px = int(round(x + distance * math.cos(mid)))
        py = int(round(y + distance * math.sin(mid)))
        if not (0 <= px < width and 0 <= py < height):
            return False
        if mask[py, px] > 0:
            return False
    return True


def interior_crossings(skel_mask, clean, points):
    """Points with four square arms on the skeleton and a facelet in each gap.

    Arms are read on the skeleton, where a merged pair of Ts reads as one
    four-armed point. The facelet probe reads the closed mask, because on a
    skeleton everything is a hole.
    """
    probe = QUAD / 100.0 * ARM_R
    kept = []
    for cx, cy in points:
        x, y = int(round(cx)), int(round(cy))
        arms = ring_arms(skel_mask, x, y, ARM_R)
        if len(arms) != 4:
            continue
        if any(abs(g - 90.0) > PERP for g in gaps_between(arms)):
            continue
        if not quadrants_dark(clean, x, y, arms, probe):
            continue
        kept.append(Cross(cx, cy, arms))
    return kept


# ======================================================= stage 4: squares
def square_corners(crossings, side_tol):
    """Every triple (p, q, r, leg) where p-q and p-r are legs of length s and
    q-r is sqrt2*s. Both tolerances are ratios, so the test is scale-free."""
    leg_tol = max(side_tol, 1) / 100.0
    hyp_tol = max(HYP_TOL, 1) / 100.0
    root2 = math.sqrt(2.0)
    n = len(crossings)
    d = [[math.hypot(crossings[i].x - crossings[j].x,
                     crossings[i].y - crossings[j].y) for j in range(n)]
         for i in range(n)]
    out = []
    for p in range(n):
        legs = [q for q in range(n) if q != p and GAP_MIN <= d[p][q] <= GAP_MAX]
        for a in range(len(legs)):
            for b in range(a + 1, len(legs)):
                q, r = legs[a], legs[b]
                lu, lv = d[p][q], d[p][r]
                if max(lu, lv) / max(min(lu, lv), 1e-9) - 1.0 > leg_tol:
                    continue
                leg = (lu + lv) / 2.0
                if abs(d[q][r] - root2 * leg) / (root2 * leg) > hyp_tol:
                    continue
                out.append((p, q, r, leg))
    return out


# ======================================================= stage 5: lattice
@dataclass
class Lattice:
    """One square of interior crossings, and what its six lines found."""

    corner: Cross                   # the triple's corner, lattice (1,1)
    u: tuple                        # step to lattice (2,1)
    v: tuple                        # step to lattice (1,2)
    fourth: tuple                   # computed (2,2), drawn, never required
    leg: float
    found: dict = field(default_factory=dict)   # (i, j) -> (x, y, "T" | "C")

    def point(self, i, j):
        return (self.corner.x + (i - 1) * self.u[0] + (j - 1) * self.v[0],
                self.corner.y + (i - 1) * self.u[1] + (j - 1) * self.v[1])

    @property
    def score(self):
        return len(self.found)


def junction_at(skel_mask, x, y, want, reach):
    """Nearest junction of the wanted kind within one disc of radius `reach`.

    want 3: a T, gaps near 90, 90 and 180.
    want 2: a corner, two arms with a right angle between them.
    One disc, nearest match wins. A search window nested inside a stepping
    loop swept hundreds of pixels per point and scored random squares as
    highly as real ones.
    """
    h, w = skel_mask.shape
    best = None
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            if dx * dx + dy * dy > reach * reach:
                continue
            px, py = int(round(x)) + dx, int(round(y)) + dy
            if not (ARM_R + 2 <= px < w - ARM_R - 2
                    and ARM_R + 2 <= py < h - ARM_R - 2):
                continue
            arms = ring_arms(skel_mask, px, py, ARM_R)
            if len(arms) != want:
                continue
            gaps = sorted(gaps_between(arms))
            if want == 3:
                ok = (abs(gaps[0] - 90.0) <= PERP and abs(gaps[1] - 90.0) <= PERP
                      and abs(gaps[2] - 180.0) <= PERP)
            else:
                ok = abs(gaps[0] - 90.0) <= PERP
            if not ok:
                continue
            d2 = dx * dx + dy * dy
            if best is None or d2 < best[0]:
                best = (d2, (float(px), float(py)))
    return best[1] if best else None


def best_lattice(skel_mask, crossings, triples, snap):
    """Anshul's construction: build each square, read along its six lines."""
    snap_frac = max(snap, 2) / 100.0
    best = None
    for p, q, r, leg in triples[:MAX_SQUARES]:
        cp, cq, cr = crossings[p], crossings[q], crossings[r]
        u = (cq.x - cp.x, cq.y - cp.y)
        v = (cr.x - cp.x, cr.y - cp.y)
        lat = Lattice(cp, u, v, (cp.x + u[0] + v[0], cp.y + u[1] + v[1]), leg)
        reach = max(int(round(snap_frac * leg)), 2)
        for i in range(4):
            for j in range(4):
                edge_i, edge_j = i in (0, 3), j in (0, 3)
                if not edge_i and not edge_j:
                    continue                    # interior; already detected
                want = 2 if (edge_i and edge_j) else 3
                gx, gy = lat.point(i, j)
                hit = junction_at(skel_mask, gx, gy, want, reach)
                if hit:
                    lat.found[(i, j)] = (hit[0], hit[1],
                                         "C" if want == 2 else "T")
        if best is None or lat.score > best.score:
            best = lat
    return best


# ================================================================ the whole
@dataclass
class Result:
    crossings: list                 # interior crossings in at least one square
    lattice: Lattice | None
    candidates: int                 # merged CN >= 3 points before any filter
    ms: float


def perceive(frame, side_tol, snap) -> Result:
    """One frame in, the face's lattice out. No state carried between frames."""
    started = time.perf_counter()
    clean = seam_mask(frame)
    skel = zhang_suen(clean)
    skel_mask = (skel * 255).astype(np.uint8)

    points = merge_points(crossing_number(skel), MERGE, ARM_R + 2)
    crossings = interior_crossings(skel_mask, clean, points)

    triples = square_corners(crossings, side_tol)
    keep = sorted({i for p, q, r, _ in triples for i in (p, q, r)})
    lattice = best_lattice(skel_mask, crossings, triples, snap)

    return Result([crossings[i] for i in keep], lattice, len(points),
                  (time.perf_counter() - started) * 1000.0)


# ================================================================== display
def dot(view, x, y, colour):
    cv2.circle(view, (int(x), int(y)), 4, colour, 1, cv2.LINE_AA)
    cv2.circle(view, (int(x), int(y)), 1, colour, -1, cv2.LINE_AA)


def draw_result(view, result, show_lines) -> None:
    for c in result.crossings:
        dot(view, c.x, c.y, CROSS_COLOUR)
    lat = result.lattice
    if lat is None:
        return
    if show_lines:
        sq = [lat.point(1, 1), lat.point(2, 1), lat.point(2, 2), lat.point(1, 2)]
        pts = [(int(round(x)), int(round(y))) for x, y in sq]
        for a in range(4):
            cv2.line(view, pts[a], pts[(a + 1) % 4], LINE_COLOUR, 1, cv2.LINE_AA)
        cv2.line(view, pts[0], pts[2], LINE_COLOUR, 1, cv2.LINE_AA)
        cv2.line(view, pts[1], pts[3], LINE_COLOUR, 1, cv2.LINE_AA)
    fx, fy = lat.fourth
    cv2.drawMarker(view, (int(round(fx)), int(round(fy))), FOURTH_COLOUR,
                   cv2.MARKER_TILTED_CROSS, 9, 1, cv2.LINE_AA)
    for x, y, kind in lat.found.values():
        dot(view, x, y, CORNER_COLOUR if kind == "C" else TEE_COLOUR)


def draw_hud(image, lines):
    """Status text top left, on a darkened panel.

    A panel rather than an outlined glyph: OpenCV's stroke fonts widen with
    thickness, so two passes at two thicknesses render visibly doubled.
    """
    if not lines:
        return
    widest = max(cv2.getTextSize(t, FONT, HUD_SCALE, 1)[0][0] for t in lines)
    panel_w = min(widest + 20, image.shape[1])
    panel_h = min(len(lines) * HUD_STEP + 10, image.shape[0])
    image[0:panel_h, 0:panel_w] //= 3
    y = 20
    for text in lines:
        cv2.putText(image, text, (10, y), FONT, HUD_SCALE,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += HUD_STEP


def hud_lines(result, params, show_lines):
    lat = result.lattice
    if lat is None:
        face = "no square"
    else:
        tees = sum(1 for _x, _y, k in lat.found.values() if k == "T")
        face = (f"leg {lat.leg:.0f} px   {tees}/8 T + {lat.score - tees}/4 C"
                f"   = {4 + lat.score}/16 lattice")
    return [
        f"{result.candidates} candidates -> {len(result.crossings)} crossings"
        f"   {result.ms:.0f} ms",
        face,
        f"side_tol {params['side_tol']}%  snap {params['snap']}%"
        f"   g lines {'on' if show_lines else 'off'}"
        "   SPACE save  i annotate  ESC quit",
    ]


# =============================================================== annotation
class Strokes:
    """Freehand annotation, kept as point lists so undo is free and the
    camera frame is never painted on."""

    def __init__(self):
        self.strokes = []
        self.drawing = False
        self.active = False
        self.erasing = False
        self.cursor = (0, 0)

    def clear(self):
        self.strokes = []
        self.drawing = False

    def undo(self):
        if self.strokes:
            self.strokes.pop()

    def erase_at(self, x, y):
        """Remove every whole stroke passing within ERASE_RADIUS of (x, y)."""
        r2 = ERASE_RADIUS * ERASE_RADIUS
        self.strokes = [s for s in self.strokes
                        if not any((px - x) ** 2 + (py - y) ** 2 <= r2
                                   for px, py in s)]

    def render_onto(self, image):
        for stroke in self.strokes:
            for a, b in zip(stroke, stroke[1:]):
                cv2.line(image, a, b, INK, INK_WIDTH, cv2.LINE_AA)
            if len(stroke) == 1:
                cv2.circle(image, stroke[0], INK_WIDTH, INK, -1, cv2.LINE_AA)


def on_mouse(event, x, y, _flags, strokes):
    if not strokes.active:
        return
    strokes.cursor = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        strokes.drawing = True
        if strokes.erasing:
            strokes.erase_at(x, y)
        else:
            strokes.strokes.append([(x, y)])
    elif event == cv2.EVENT_MOUSEMOVE and strokes.drawing:
        if strokes.erasing:
            strokes.erase_at(x, y)
        else:
            strokes.strokes[-1].append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        strokes.drawing = False


def annotate(display, frame, strokes, capture_dir) -> int:
    """Freeze, draw freehand, save the camera frame plus a `_marked` copy."""
    strokes.clear()
    strokes.erasing = False
    strokes.active = True
    try:
        while True:
            view = display.copy()
            strokes.render_onto(view)
            if strokes.erasing:
                cv2.circle(view, strokes.cursor, ERASE_RADIUS, INK, 1, cv2.LINE_AA)
            draw_hud(view, [
                "ERASE  drag to rub out" if strokes.erasing else "DRAW  drag to draw",
                f"strokes: {len(strokes.strokes)}",
                "SPACE save both   e draw/erase   u undo   c clear   ESC discard",
            ])
            cv2.imshow(WINDOW, view)

            key = cv2.waitKey(1) & 0xFF
            if key == KEY_ESC:
                print("discarded")
                return 0
            if key == ord("e"):
                strokes.erasing = not strokes.erasing
                strokes.drawing = False
            elif key == ord("u"):
                strokes.undo()
            elif key == ord("c"):
                strokes.clear()
            elif key == KEY_SPACE:
                stem = f"perception_{timestamped_stem()}"
                marked = display.copy()
                strokes.render_onto(marked)
                return (write_png(capture_dir / f"{stem}.png", frame)
                        + write_png(capture_dir / f"{stem}_marked.png", marked))
    finally:
        strokes.active = False


# ==================================================================== plumbing
def timestamped_stem() -> str:
    now = datetime.now()
    return f"{now:%Y-%m-%d_%H%M%S}_{now.microsecond // 1000:03d}"


def write_png(path, image) -> bool:
    if cv2.imwrite(str(path), image):
        print(f"saved {path.name}")
        return True
    print(f"ERROR: could not write {path}", file=sys.stderr)
    return False


def report_opencv_origin() -> None:
    """Print which OpenCV loaded. A ROS-sourced shell exports PYTHONPATH, which
    a venv honours, and the system OpenCV it brings in cannot open this camera.
    That looks like a dead camera unless the import says where it came from."""
    origin = Path(cv2.__file__).resolve()
    print(f"opencv {cv2.__version__} from {origin.parent}")
    if not origin.is_relative_to(Path(sys.prefix).resolve()):
        print("WARNING: that OpenCV is not from this venv. If the camera will "
              "not open, clear PYTHONPATH and try again.", file=sys.stderr)


def read_trackbars() -> dict:
    return {name: cv2.getTrackbarPos(name, CONTROLS) for name, _, _ in TRACKBARS}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find the 4x4 lattice of a cube face in a webcam stream.")
    parser.add_argument("camera", nargs="?", type=int, default=0,
                        help="camera index (default 0)")
    parser.add_argument("--dir", type=Path, default=DEFAULT_CAPTURE_DIR,
                        help=f"where images are written (default {DEFAULT_CAPTURE_DIR})")
    args = parser.parse_args()

    capture_dir = args.dir.expanduser().resolve()
    capture_dir.mkdir(parents=True, exist_ok=True)
    report_opencv_origin()

    cam = cv2.VideoCapture(args.camera)
    if not cam.isOpened():
        print(f"ERROR: could not open camera {args.camera}.", file=sys.stderr)
        return 1
    print(f"camera {args.camera} open at "
          f"{int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))}, writing to {capture_dir}")

    cv2.namedWindow(WINDOW)
    cv2.namedWindow(CONTROLS, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROLS, 400, 120)
    for name, default, maximum in TRACKBARS:
        cv2.createTrackbar(name, CONTROLS, default, maximum, lambda _v: None)

    strokes = Strokes()
    cv2.setMouseCallback(WINDOW, on_mouse, strokes)
    show_lines = False
    saved = 0

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                print("ERROR: camera stopped delivering frames.", file=sys.stderr)
                return 1

            params = read_trackbars()
            result = perceive(frame, params["side_tol"], params["snap"])

            shown = frame.copy()
            draw_result(shown, result, show_lines)
            draw_hud(shown, hud_lines(result, params, show_lines))
            cv2.imshow(WINDOW, shown)

            key = cv2.waitKey(1) & 0xFF
            if key == KEY_ESC:
                break
            if key == ord("g"):
                show_lines = not show_lines
            elif key == KEY_SPACE:
                stem = f"perception_{timestamped_stem()}"
                saved += write_png(capture_dir / f"{stem}.png", frame)
                saved += write_png(capture_dir / f"{stem}_view.png", shown)
            elif key == ord("i"):
                saved += annotate(shown, frame, strokes, capture_dir)
    finally:
        cam.release()
        cv2.destroyAllWindows()

    print(f"{saved} file(s) in {capture_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
