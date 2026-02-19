from __future__ import annotations

import random
from typing import List

from agents.observation import Observation
from core.types import ACTION_DELTAS


class GreedyAgent:
    # Moves to reduce Manhattan distance when possible.
    name = "greedy"

    def choose_action(self, obs: Observation, prompt: str | None = None) -> str:
        best_actions: List[str] = []
        best_dist = None
        for action, is_open in obs.open_map.items():
            if not is_open:
                continue
            dx, dy = ACTION_DELTAS[action]
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
