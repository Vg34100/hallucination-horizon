from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from core.types import Coord


@dataclass
class Observation:
    pos: Coord
    goal: Coord
    open_map: Dict[str, bool]
    last_action: str | None
    step: int
