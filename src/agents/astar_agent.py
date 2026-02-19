from __future__ import annotations

from typing import List

from agents.observation import Observation
from core.types import ACTION_DELTAS, Coord


class AStarAgent:
    # Follows a precomputed A* path (reliable baseline).
    name = "astar"

    def __init__(self, path: List[Coord]) -> None:
        self.path = path
        self.idx = 0

    def choose_action(self, obs: Observation, prompt: str | None = None) -> str:
        if not self.path or self.idx >= len(self.path) - 1:
            return "N"
        if obs.pos == self.path[self.idx]:
            next_pos = self.path[self.idx + 1]
        else:
            # resync if off path
            try:
                self.idx = self.path.index(obs.pos)
                if self.idx >= len(self.path) - 1:
                    return "N"
                next_pos = self.path[self.idx + 1]
            except ValueError:
                return "N"
        dx = next_pos[0] - obs.pos[0]
        dy = next_pos[1] - obs.pos[1]
        action = {
            (0, -1): "N",
            (0, 1): "S",
            (1, 0): "E",
            (-1, 0): "W",
        }.get((dx, dy), "N")
        self.idx += 1
        return action
