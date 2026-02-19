from __future__ import annotations

import random
from typing import List, Tuple

from core.types import Coord


def make_simple_maze() -> Tuple[int, int, List[Coord], Coord, Coord]:
    # Small maze for quick tests.
    width, height = 7, 7
    walls = {
        (1, 1), (2, 1), (3, 1), (5, 1),
        (1, 2), (5, 2),
        (1, 3), (3, 3), (4, 3), (5, 3),
        (1, 4),
        (3, 5), (4, 5), (5, 5),
    }
    start = (0, 0)
    goal = (6, 6)
    return width, height, list(walls), start, goal


def make_hard_maze() -> Tuple[int, int, List[Coord], Coord, Coord]:
    # Winding corridor maze (harder, no trivial straight path).
    width, height = 9, 9
    start = (0, 0)
    goal = (8, 8)

    path = [
        (0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0),
        (3, 0), (4, 0), (4, 1), (4, 2), (4, 3), (3, 3), (2, 3),
        (2, 4), (2, 5), (3, 5), (4, 5), (5, 5), (6, 5),
        (6, 4), (6, 3), (7, 3), (8, 3), (8, 4), (8, 5),
        (8, 6), (7, 6), (6, 6), (6, 7), (6, 8), (7, 8), (8, 8),
    ]
    open_set = set(path)
    walls: List[Coord] = []
    for y in range(height):
        for x in range(width):
            if (x, y) not in open_set:
                walls.append((x, y))
    return width, height, walls, start, goal


def make_random_maze(width: int, height: int, wall_prob: float = 0.25, seed: int | None = None) -> Tuple[int, int, List[Coord], Coord, Coord]:
    # Simple random maze generator (not guaranteed solvable).
    if seed is not None:
        random.seed(seed)
    start = (0, 0)
    goal = (width - 1, height - 1)
    walls: List[Coord] = []
    for y in range(height):
        for x in range(width):
            if (x, y) in (start, goal):
                continue
            if random.random() < wall_prob:
                walls.append((x, y))
    return width, height, walls, start, goal
