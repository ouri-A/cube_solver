"""Face-turn notation, checked before a goal is accepted. No ROS in here."""

import re

MOVE = re.compile(r"[URFDLB]['2]?")   # a quarter turn clockwise, anticlockwise ('), or half (2)


def bad_moves(moves):
    """Return the moves that are not a single face turn, in the order given."""
    return [m for m in moves if not MOVE.fullmatch(m)]
