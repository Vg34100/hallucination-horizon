from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from core.types import Coord


@dataclass
class MetricsState:
    visit_counts: Dict[Coord, int] = field(default_factory=dict)
    loop_events: int = 0
    invalid_moves: int = 0
    steps: int = 0

    def update_visit(self, pos: Coord, loop_threshold: int = 3) -> bool:
        count = self.visit_counts.get(pos, 0) + 1
        self.visit_counts[pos] = count
        if count == loop_threshold:
            self.loop_events += 1
            return True
        return False

    def update_invalid(self, is_invalid: bool) -> None:
        if is_invalid:
            self.invalid_moves += 1

    def rate_invalid(self) -> float:
        return self.invalid_moves / self.steps if self.steps else 0.0

    def rate_loops(self) -> float:
        return self.loop_events / self.steps if self.steps else 0.0
