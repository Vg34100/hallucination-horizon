from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

Coord = Tuple[int, int]


@dataclass
class Observation:
    pos: Coord
    goal: Coord
    open_map: Dict[str, bool]
    last_action: str | None
    step: int


class RandomAgent:
    name = "random"

    def choose_action(self, obs: Observation, prompt: str | None = None) -> str:
        actions = [a for a, is_open in obs.open_map.items() if is_open]
        if not actions:
            return "N"
        return random.choice(actions)


class GreedyAgent:
    name = "greedy"

    def choose_action(self, obs: Observation, prompt: str | None = None) -> str:
        best_actions: List[str] = []
        best_dist = None
        for action, is_open in obs.open_map.items():
            if not is_open:
                continue
            dx, dy = {
                "N": (0, -1),
                "S": (0, 1),
                "E": (1, 0),
                "W": (-1, 0),
            }[action]
            next_pos = (obs.pos[0] + dx, obs.pos[1] + dy)
            dist = abs(next_pos[0] - obs.goal[0]) + abs(next_pos[1] - obs.goal[1])
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_actions = [action]
            elif dist == best_dist:
                best_actions.append(action)
        if not best_actions:
            return "N"
        return random.choice(best_actions)


class AStarAgent:
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
