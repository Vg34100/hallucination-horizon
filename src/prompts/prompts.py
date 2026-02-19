from __future__ import annotations

from typing import Dict, List

from core.types import Coord


def format_observation(
    pos: Coord,
    goal: Coord,
    open_map: Dict[str, bool],
    last_action: str | None,
    step: int,
) -> str:
    def status(action: str) -> str:
        return "open" if open_map.get(action, False) else "wall"

    last_action_str = last_action or "None"
    return (
        f"You are at {pos}. Goal is at {goal}.\n"
        f"North: {status('N')}. South: {status('S')}. East: {status('E')}. West: {status('W')}.\n"
        f"Last move: {last_action_str}. Step: {step}."
    )


def format_local_grid(
    pos: Coord,
    goal: Coord,
    walls: set[Coord],
    width: int,
    height: int,
    radius: int,
) -> str:
    lines: List[str] = []
    for y in range(pos[1] - radius, pos[1] + radius + 1):
        row = []
        for x in range(pos[0] - radius, pos[0] + radius + 1):
            if x < 0 or y < 0 or x >= width or y >= height:
                row.append(" ")
            elif (x, y) == pos:
                row.append("A")
            elif (x, y) == goal:
                row.append("G")
            elif (x, y) in walls:
                row.append("#")
            else:
                row.append(".")
        lines.append("".join(row))
    return "\n".join(lines)


def build_prompt(
    obs_text: str,
    history: List[str],
    local_grid: str | None,
) -> str:
    parts: List[str] = []
    parts.append("You are navigating a grid maze to reach the goal.")
    parts.append("Think step by step: consider position, goal, available moves, and recent history.")
    if history:
        parts.append("Recent history (most recent last):")
        parts.extend(history)
    if local_grid:
        parts.append("Local grid (A=agent, G=goal, #=wall, .=open):")
        parts.append(local_grid)
    parts.append("Current observation:")
    parts.append(obs_text)
    parts.append("What is the best move toward the goal? Respond with one letter: N, S, E, or W.")
    return "\n".join(parts)
