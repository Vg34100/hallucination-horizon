from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple


Coord = Tuple[int, int]


ACTION_DELTAS = {
    "N": (0, -1),
    "S": (0, 1),
    "E": (1, 0),
    "W": (-1, 0),
}


@dataclass
class StepResult:
    pos_before: Coord
    pos_after: Coord
    action: str
    valid_move: bool
    hit_wall: bool
    out_of_bounds: bool


class GridEnv:
    def __init__(
        self,
        width: int,
        height: int,
        walls: Iterable[Coord],
        start: Coord,
        goal: Coord,
        flip_candidates: Iterable[Coord] | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.walls: Set[Coord] = set(walls)
        self.start = start
        self.goal = goal
        self.pos = start
        if flip_candidates is None:
            self.flip_candidates = []
        else:
            self.flip_candidates = list(flip_candidates)

    def reset(self) -> Coord:
        self.pos = self.start
        return self.pos

    def in_bounds(self, pos: Coord) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def is_wall(self, pos: Coord) -> bool:
        return pos in self.walls

    def neighbors_open(self, pos: Coord) -> Dict[str, bool]:
        open_map: Dict[str, bool] = {}
        for action, (dx, dy) in ACTION_DELTAS.items():
            nx, ny = pos[0] + dx, pos[1] + dy
            npos = (nx, ny)
            open_map[action] = self.in_bounds(npos) and not self.is_wall(npos)
        return open_map

    def step(self, action: str) -> StepResult:
        action = action.strip().upper()
        if action not in ACTION_DELTAS:
            return StepResult(
                pos_before=self.pos,
                pos_after=self.pos,
                action=action,
                valid_move=False,
                hit_wall=False,
                out_of_bounds=False,
            )

        dx, dy = ACTION_DELTAS[action]
        pos_before = self.pos
        next_pos = (pos_before[0] + dx, pos_before[1] + dy)

        out_of_bounds = not self.in_bounds(next_pos)
        hit_wall = self.is_wall(next_pos)
        valid_move = not out_of_bounds and not hit_wall

        if valid_move:
            self.pos = next_pos

        return StepResult(
            pos_before=pos_before,
            pos_after=self.pos,
            action=action,
            valid_move=valid_move,
            hit_wall=hit_wall,
            out_of_bounds=out_of_bounds,
        )

    def flip_wall(self) -> Coord | None:
        if not self.flip_candidates:
            return None
        cell = random.choice(self.flip_candidates)
        if cell in self.walls:
            self.walls.remove(cell)
        else:
            self.walls.add(cell)
        return cell


def make_simple_maze() -> Tuple[int, int, List[Coord], Coord, Coord]:
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
    width, height = 9, 9
    start = (0, 0)
    goal = (8, 8)

    # Define a single winding corridor path (no trivial S->E loop).
    path = [
        (0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0),
        (3, 0), (4, 0), (4, 1), (4, 2), (4, 3), (3, 3), (2, 3),
        (2, 4), (2, 5), (3, 5), (4, 5), (5, 5), (6, 5),
        (6, 4), (6, 3), (7, 3), (8, 3), (8, 4), (8, 5),
        (8, 6), (7, 6), (6, 6), (6, 7), (6, 8), (7, 8), (8, 8),
    ]
    open_set = set(path)
    walls = []
    for y in range(height):
        for x in range(width):
            if (x, y) not in open_set:
                walls.append((x, y))
    return width, height, walls, start, goal
