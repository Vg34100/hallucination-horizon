from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set

from core.types import ACTION_DELTAS, Coord


@dataclass
class StepResult:
    pos_before: Coord
    pos_after: Coord
    action: str
    valid_move: bool
    hit_wall: bool
    out_of_bounds: bool


class GridEnv:
    # Simple gridworld: walls are blocked cells, agent has N/S/E/W actions.
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
        # Returns which of N/S/E/W are open from current position.
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
        # Toggle one candidate wall cell on/off (for dynamic environments).
        if not self.flip_candidates:
            return None
        cell = random.choice(self.flip_candidates)
        if cell in self.walls:
            self.walls.remove(cell)
        else:
            self.walls.add(cell)
        return cell
