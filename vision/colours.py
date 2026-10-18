"""Each face letter's display colour, from what a scan actually read.

Used by the solve viewer (from a saved scan.json) and by the scan itself (from the
readings still in memory), so both paint a sticker in the colour the camera saw.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from cube_model import FACES
from sample import lab_to_bgr


def face_colours(lab_means: Mapping[str, np.ndarray], state: str) -> dict[str, tuple[float, float, float]]:
    """RGB 0-1 per face letter: the mean LAB of every sticker the state gives that letter.

    lab_means maps each face in FACES to its nine stickers' mean LAB, shape (9, 3), in the
    same sticker order as state.
    """
    lab = np.concatenate([np.asarray(lab_means[f], float) for f in FACES])
    letters = np.array(list(state))
    out = {}
    for f in FACES:
        b, g, r = lab_to_bgr(lab[letters == f].mean(axis=0))
        out[f] = (r / 255, g / 255, b / 255)
    return out
