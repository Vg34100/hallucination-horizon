from __future__ import annotations

import random

from agents.observation import Observation


class RandomAgent:
    # Picks any currently open direction at random.
    name = "random"

    def choose_action(self, obs: Observation, prompt: str | None = None) -> str:
        actions = [a for a, is_open in obs.open_map.items() if is_open]
        if not actions:
            return "N"
        return random.choice(actions)
