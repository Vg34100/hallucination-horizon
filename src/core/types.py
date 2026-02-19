from __future__ import annotations

from typing import Tuple

Coord = Tuple[int, int]

ACTION_DELTAS = {
    "N": (0, -1),
    "S": (0, 1),
    "E": (1, 0),
    "W": (-1, 0),
}
