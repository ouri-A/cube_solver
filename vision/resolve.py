#!/usr/bin/env python3
"""
Sticker colours -> kociemba state, by matching whole pieces.

dwalton76's method (rubiks-color-resolver), our code. No colour is ever
named: the six centres are the reference colours, and every sticker gets the
letter of a face.

    1. References    each face's centre, the MEDIAN of its patch, so a logo
                     covering less than half the patch does not move it
    2. Corners       the 8 real corners (URF, UFL, ...) and the 8 scanned ones
                     are cities in a travelling-salesman problem. Real-to-
                     scanned distance = summed LAB distance of the three
                     stickers in the best of the 3 rotations; same-kind = huge,
                     so the shortest route zigzags real, scanned, real, ... and
                     consecutive pairs are matches. Then a swap pass: swap the
                     scanned pieces of two matches while that lowers the total
    3. Edges         the same with the 12 edges and 2 orientations
    4. Refine        each reference becomes the mean of the non-centre stickers
                     it was given (a big logo is left out entirely), then match
                     again, until the state stops changing
    5. Check         colour counts, each piece once, twist, flip, parity

Every real piece is used exactly once, so a misread sticker is outvoted by
the other stickers on its piece.

    uv run resolve.py scans/<scan>.json      re-resolve a saved scan
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cube_model import CENTRES, CORNER_SLOTS, CORNERS, EDGE_SLOTS, EDGES, FACES, Check, check

SAME_KIND = 1e6          # TSP distance between two real or two scanned pieces
MAX_ROUNDS = 5           # match / refine rounds before giving up on convergence


@dataclass
class Resolution:
    state: str                   # 54 letters, kociemba order
    check: Check
    rounds: int                  # match / refine rounds run
    piece_cost: dict[str, float]  # slot name -> summed LAB distance of its match
    refs: dict[str, list[float]]  # final reference LAB per face

    @property
    def worst(self) -> tuple[str, float]:
        return max(self.piece_cost.items(), key=lambda kv: kv[1])


# ------------------------------------------------------------------- costs
def piece_costs(lab: np.ndarray, refs: dict[str, np.ndarray],
                slots: list[tuple[int, ...]], pieces: list[str]):
    """cost[s, p] and turn[s, p]: best summed LAB distance of scanned slot s
    to real piece p, and the rotation that achieves it. Under rotation t,
    sticker i of the slot carries the colour of face pieces[p][(i + t) % k]."""
    k = len(pieces[0])
    cost = np.empty((len(slots), len(pieces)))
    turn = np.empty((len(slots), len(pieces)), int)
    for s, slot in enumerate(slots):
        for p, piece in enumerate(pieces):
            c = [sum(np.linalg.norm(lab[f] - refs[piece[(i + t) % k]]) for i, f in enumerate(slot))
                 for t in range(k)]
            turn[s, p] = int(np.argmin(c))
            cost[s, p] = c[turn[s, p]]
    return cost, turn


# --------------------------------------------------------------------- TSP
def greedy_route(D: np.ndarray) -> list[int]:
    """Open route through every node: take the shortest links first, never
    giving a node a third link or closing a loop (greedy edge matching)."""
    n = len(D)
    parent = list(range(n))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    degree, links = [0] * n, [[] for _ in range(n)]
    pairs = sorted(((D[i, j], i, j) for i in range(n) for j in range(i + 1, n)))
    added = 0
    for _, i, j in pairs:
        if degree[i] < 2 and degree[j] < 2 and root(i) != root(j):
            parent[root(i)] = root(j)
            degree[i] += 1
            degree[j] += 1
            links[i].append(j)
            links[j].append(i)
            added += 1
            if added == n - 1:
                break
    start = next(i for i in range(n) if degree[i] < 2)
    route, prev = [start], None
    while len(route) < n:
        nxt = next(j for j in links[route[-1]] if j != prev)
        prev = route[-1]
        route.append(nxt)
    return route


def two_opt(route: list[int], D: np.ndarray) -> list[int]:
    """Reverse a stretch of the route while that shortens it."""
    route, n = route[:], len(route)
    improved = True
    while improved:
        improved = False
        for a in range(n - 3):
            for c in range(a + 2, n - 1):
                p, q, r, s = route[a], route[a + 1], route[c], route[c + 1]
                if D[p, q] + D[r, s] > D[p, r] + D[q, s] + 1e-9:
                    route[a + 1:c + 1] = route[a + 1:c + 1][::-1]
                    improved = True
    return route


def match_tsp(cost: np.ndarray) -> list[int]:
    """slot -> piece, one-to-one, by the zigzag route plus the swap pass."""
    m = len(cost)                       # nodes 0..m-1 real pieces, m..2m-1 scanned slots
    D = np.full((2 * m, 2 * m), SAME_KIND)
    D[:m, m:] = cost.T
    D[m:, :m] = cost
    np.fill_diagonal(D, 0)
    route = two_opt(greedy_route(D), D)

    piece_of = [-1] * m
    used = set()
    i = 0
    while i < len(route) - 1:           # consecutive real/scanned neighbours are matches
        a, b = route[i], route[i + 1]
        if (a < m) != (b < m):
            p, s = (a, b - m) if a < m else (b, a - m)
            piece_of[s] = p
            used.add(p)
            i += 2
        else:
            i += 1
    for s in range(m):                  # route broke the zigzag somewhere: cheapest leftovers
        if piece_of[s] < 0:
            p = min((p for p in range(m) if p not in used), key=lambda p: cost[s, p])
            piece_of[s] = p
            used.add(p)

    while True:                         # swap pass: best improving swap, until none
        best, pair = 1e-9, None
        for x in range(m):
            for y in range(x + 1, m):
                px, py = piece_of[x], piece_of[y]
                delta = cost[x, px] + cost[y, py] - cost[x, py] - cost[y, px]
                if delta > best:
                    best, pair = delta, (x, y)
        if pair is None:
            return piece_of
        x, y = pair
        piece_of[x], piece_of[y] = piece_of[y], piece_of[x]


def match_assignment(cost: np.ndarray) -> list[int]:
    """The same matching solved exactly (Hungarian algorithm). For comparison."""
    from scipy.optimize import linear_sum_assignment
    _, piece_of = linear_sum_assignment(cost)
    return [int(p) for p in piece_of]


# ----------------------------------------------------------------- resolve
def resolve(lab: np.ndarray, centre_refs: dict[str, np.ndarray], method: str = "tsp") -> Resolution:
    """lab: (54, 3) mean LAB per sticker in kociemba order.
    centre_refs: face letter -> median LAB of that face's centre patch."""
    match = {"tsp": match_tsp, "assignment": match_assignment}[method]
    refs = {f: np.asarray(centre_refs[f], float) for f in FACES}
    state, rounds = None, 0
    while rounds < MAX_ROUNDS:
        rounds += 1
        letters = [""] * 54
        piece_cost = {}
        for f, i in zip(FACES, CENTRES):
            letters[i] = f
        for slots, pieces in ((CORNER_SLOTS, CORNERS), (EDGE_SLOTS, EDGES)):
            cost, turn = piece_costs(lab, refs, slots, pieces)
            for s, p in enumerate(match(cost)):
                k = len(pieces[p])
                for i, f in enumerate(slots[s]):
                    letters[f] = pieces[p][(i + turn[s, p]) % k]
                piece_cost[pieces[s]] = float(cost[s, p])
        new = "".join(letters)
        if new == state:
            break
        state = new
        for f in FACES:                 # refine: references from the stickers each face was given
            members = [i for i in range(54) if letters[i] == f and i not in CENTRES]
            refs[f] = lab[members].mean(0)
    return Resolution(state, check(state), rounds, piece_cost, {f: refs[f].tolist() for f in FACES})


# ------------------------------------------------------------------- scans
def from_scan(scan: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """(54, 3) sticker LAB and centre references from a saved scan JSON."""
    lab = np.zeros((54, 3))
    refs = {}
    for k, f in enumerate(FACES):
        face = scan["faces"][f]
        lab[9 * k:9 * k + 9] = np.asarray(face["lab_mean"], float)
        refs[f] = np.asarray(face["lab_median"][4], float)
    return lab, refs


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-resolve a saved scan.")
    parser.add_argument("scan", type=Path)
    parser.add_argument("--method", choices=["tsp", "assignment"], default="tsp")
    args = parser.parse_args()
    lab, refs = from_scan(json.loads(args.scan.read_text()))
    res = resolve(lab, refs, args.method)
    print(res.state)
    print("legal" if res.check.ok else "ILLEGAL: " + "; ".join(res.check.problems))
    print(f"rounds {res.rounds}, worst piece {res.worst[0]} at {res.worst[1]:.1f}")
    return 0 if res.check.ok else 1


if __name__ == "__main__":
    sys.exit(main())
