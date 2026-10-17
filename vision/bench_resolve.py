#!/usr/bin/env python3
"""
Benchmark the resolver on cubes built from REAL readings of your cube.

Scan a SOLVED cube with scan.py once per lighting (window light, room
lights). Each scan gives 9 real readings per colour. A test cube is a random
scramble whose every sticker takes a reading of its colour drawn from ONE
scan, so all 54 stickers share that scan's lighting; the centre references
are that scan's real centre medians. The resolver must give back the exact
scramble.

    uv run bench_resolve.py scans/<window>/scan.json scans/<lights>/scan.json
    uv run bench_resolve.py scans/*/scan.json --cubes 2000

Reported per scan: right, wrong-but-legal (the dangerous kind: it would be
solved and the solution would be wrong), and illegal (caught by the checks),
for the TSP matching and for the exact Hungarian matching.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

from cube_model import CENTRES, FACES, SOLVED, apply_moves, random_moves
from resolve import resolve


def pools(scan: dict) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Per face letter: its 9 real mean readings, and its centre median."""
    readings = {f: np.asarray(scan["faces"][f]["lab_mean"], float) for f in FACES}
    centres = {f: np.asarray(scan["faces"][f]["lab_median"][4], float) for f in FACES}
    return readings, centres


def bench(scan: dict, cubes: int, seed: int) -> dict[str, dict[str, int]]:
    readings, centres = pools(scan)
    rng = random.Random(seed)
    out = {m: {"right": 0, "wrong_but_legal": 0, "illegal": 0} for m in ("tsp", "assignment")}
    for _ in range(cubes):
        state = apply_moves(SOLVED, random_moves(25, rng))
        lab = np.array([readings[c][rng.randrange(9)] for c in state])
        for f, i in zip(FACES, CENTRES):
            lab[i] = readings[f][4]
        for method in out:
            res = resolve(lab, centres, method)
            key = "right" if res.state == state else "wrong_but_legal" if res.check.ok else "illegal"
            out[method][key] += 1
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolver benchmark on real solved-cube scans.")
    parser.add_argument("scans", nargs="+", type=Path, help="scan.json of a SOLVED cube, one per lighting")
    parser.add_argument("--cubes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for path in args.scans:
        scan = json.loads(path.read_text())
        if scan.get("state") not in (None, SOLVED):
            print(f"{path}: WARNING, this scan did not read as solved ({scan.get('state')})")
        result = bench(scan, args.cubes, args.seed)
        print(f"{path}  ({args.cubes} cubes)")
        for method, r in result.items():
            print(f"  {method:10s}  right {r['right']:5d}   wrong-but-legal {r['wrong_but_legal']:4d}"
                  f"   illegal {r['illegal']:4d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
