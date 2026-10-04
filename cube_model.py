"""
The cube at the sticker level, in kociemba's layout.

A state is a 54-letter string over U R F D L B: the letter names the face whose
centre has that sticker's colour. Order is U1..U9, R1..R9, F1..F9, D1..D9,
L1..L9, B1..B9, each face read row by row as seen from outside with the net
orientation below (the same string kociemba.solve takes).

                 U1 U2 U3
                 U4 U5 U6
                 U7 U8 U9
        L1 L2 L3 F1 F2 F3 R1 R2 R3 B1 B2 B3
        L4 L5 L6 F4 F5 F6 R4 R5 R6 B4 B5 B6
        L7 L8 L9 F7 F8 F9 R7 R8 R9 B7 B8 B9
                 D1 D2 D3
                 D4 D5 D6
                 D7 D8 D9

Holding the cube with F towards you and U on top, each face's row-major order
is exactly what a camera sees when that face is turned to it by the guided
scan: R, B, L by turning the cube left, U by tipping the top towards you, D by
tipping the top away.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

FACES = "URFDLB"
SOLVED = "".join(f * 9 for f in FACES)


def index(name: str) -> int:
    """'U1' -> 0, 'B9' -> 53."""
    return 9 * FACES.index(name[0]) + int(name[1]) - 1


def _indices(rows: str) -> list[tuple[int, ...]]:
    return [tuple(index(n) for n in group.split()) for group in rows.split(",")]


# Each corner and edge slot as its stickers, in kociemba's order (pykociemba
# facecube.py). The first sticker of every slot is on U or D, or for the four
# middle-layer edges on F or B; that sticker is what orientation is counted from.
# Corners list their stickers clockwise.
CORNER_SLOTS = _indices("U9 R1 F3, U7 F1 L3, U1 L1 B3, U3 B1 R3, D3 F9 R7, D1 L9 F7, D7 B9 L7, D9 R9 B7")
EDGE_SLOTS = _indices("U6 R2, U8 F2, U4 L2, U2 B2, D6 R8, D2 F8, D4 L8, D8 B8, F6 R4, F4 L6, B6 L4, B4 R6")
# The real pieces, named by the faces whose colours they carry, in the same order.
CORNERS = ["URF", "UFL", "ULB", "UBR", "DFR", "DLF", "DBL", "DRB"]
EDGES = ["UR", "UF", "UL", "UB", "DR", "DF", "DL", "DB", "FR", "FL", "BL", "BR"]
CENTRES = [index(f + "5") for f in FACES]


# ------------------------------------------------------------------ moves
# Every sticker as (cubie position, outward normal) in a frame with x towards
# R, y towards U, z towards F. Per face: normal, then the image directions of
# increasing column and increasing row in the net above.
_AXES = {
    "U": ((0, 1, 0), (1, 0, 0), (0, 0, 1)),
    "R": ((1, 0, 0), (0, 0, -1), (0, -1, 0)),
    "F": ((0, 0, 1), (1, 0, 0), (0, -1, 0)),
    "D": ((0, -1, 0), (1, 0, 0), (0, 0, -1)),
    "L": ((-1, 0, 0), (0, 0, 1), (0, -1, 0)),
    "B": ((0, 0, -1), (-1, 0, 0), (0, -1, 0)),
}


def _stickers() -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    out = []
    for f in FACES:
        n, right, down = (np.array(v) for v in _AXES[f])
        for r in range(3):
            for c in range(3):
                p = n + right * (c - 1) + down * (r - 1)
                out.append((tuple(int(x) for x in p), tuple(int(x) for x in n)))
    return out


_STICKERS = _stickers()
_WHERE = {s: k for k, s in enumerate(_STICKERS)}


def _quarter_turn(face: str) -> list[int]:
    """Permutation for a clockwise quarter turn of `face`: new[j] = old[perm[j]]."""
    a = np.array(_AXES[face][0])

    def rot(v):  # -90 degrees about a: clockwise seen from outside the face
        v = np.array(v)
        return tuple(int(x) for x in a * (a @ v) - np.cross(a, v))

    perm = list(range(54))
    for i, (p, n) in enumerate(_STICKERS):
        if np.dot(p, a) == 1:
            perm[_WHERE[(rot(p), rot(n))]] = i
    return perm


QUARTER = {f: _quarter_turn(f) for f in FACES}


def apply_moves(state: str, moves: str) -> str:
    """Apply moves in standard notation, e.g. "R U2 F'"."""
    s = list(state)
    for m in moves.split():
        turns = {"": 1, "2": 2, "'": 3}[m[1:]]
        for _ in range(turns):
            s = [s[i] for i in QUARTER[m[0]]]
    return "".join(s)


def random_moves(n: int, rng: random.Random) -> str:
    """n random face turns, never the same face twice in a row."""
    out, last = [], None
    for _ in range(n):
        f = rng.choice([x for x in FACES if x != last])
        out.append(f + rng.choice(["", "2", "'"]))
        last = f
    return " ".join(out)


# ----------------------------------------------------------------- checks
@dataclass
class Check:
    ok: bool
    problems: list[str] = field(default_factory=list)


def _parity(perm: list[int]) -> int:
    """0 if perm is an even permutation, 1 if odd."""
    seen, swaps = [False] * len(perm), 0
    for i in range(len(perm)):
        j, length = i, 0
        while not seen[j]:
            seen[j], j, length = True, perm[j], length + 1
        swaps += max(length - 1, 0)
    return swaps % 2


def check(state: str) -> Check:
    """Every condition a real cube satisfies: colour counts, each piece once,
    corner twist, edge flip, and permutation parity. Same tests as kociemba's
    verify, but reporting which one failed."""
    problems = []
    for f in FACES:
        if state.count(f) != 9:
            problems.append(f"{state.count(f)} stickers of {f}, need 9")
    for f, i in zip(FACES, CENTRES):
        if state[i] != f:
            problems.append(f"centre of {f} reads {state[i]}")
    if problems:
        return Check(False, problems)

    corner_perm, twist = [], 0
    for k, slot in enumerate(CORNER_SLOTS):
        cols = "".join(state[i] for i in slot)
        ori = next((o for o in range(3) if cols[o] in "UD"), None)
        piece = None if ori is None else cols[ori:] + cols[:ori]
        if piece not in CORNERS:
            problems.append(f"corner slot {CORNERS[k]} reads {cols}, not a real corner")
            continue
        corner_perm.append(CORNERS.index(piece))
        twist += ori
    edge_perm, flip = [], 0
    for k, slot in enumerate(EDGE_SLOTS):
        cols = "".join(state[i] for i in slot)
        if cols in EDGES:
            edge_perm.append(EDGES.index(cols))
        elif cols[::-1] in EDGES:
            edge_perm.append(EDGES.index(cols[::-1]))
            flip += 1
        else:
            problems.append(f"edge slot {EDGES[k]} reads {cols}, not a real edge")
    if problems:
        return Check(False, problems)

    if len(set(corner_perm)) != 8:
        problems.append("a corner appears twice")
    if len(set(edge_perm)) != 12:
        problems.append("an edge appears twice")
    if problems:
        return Check(False, problems)
    if twist % 3:
        problems.append(f"corner twists sum to {twist}, not a multiple of 3")
    if flip % 2:
        problems.append(f"{flip} edges flipped, need an even number")
    if _parity(corner_perm) != _parity(edge_perm):
        problems.append("corner and edge permutation parity differ")
    return Check(not problems, problems)
