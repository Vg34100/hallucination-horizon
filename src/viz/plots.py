from __future__ import annotations

from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np

from core.types import Coord


def _grid_to_image(width: int, height: int, walls: Iterable[Coord]) -> np.ndarray:
    grid = np.zeros((height, width))
    for x, y in walls:
        grid[y, x] = 1
    return grid


def plot_trajectory(
    width: int,
    height: int,
    walls: Iterable[Coord],
    start: Coord,
    goal: Coord,
    astar_path: List[Coord],
    agent_path: List[Coord],
    out_path: str,
) -> None:
    grid = _grid_to_image(width, height, walls)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(grid, cmap="Greys", origin="upper")

    if astar_path:
        xs, ys = zip(*astar_path)
        ax.plot(xs, ys, color="blue", linewidth=2, label="A* path")
    if agent_path:
        xs, ys = zip(*agent_path)
        ax.plot(xs, ys, color="red", linewidth=2, label="Agent path")

    ax.scatter([start[0]], [start[1]], color="green", s=60, label="Start")
    ax.scatter([goal[0]], [goal[1]], color="gold", s=60, label="Goal")
    ax.set_xticks(range(width))
    ax.set_yticks(range(height))
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.grid(True, linewidth=0.5, alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_heatmap(
    width: int,
    height: int,
    counts: Dict[Coord, int],
    title: str,
    out_path: str,
) -> None:
    grid = np.zeros((height, width))
    for (x, y), v in counts.items():
        grid[y, x] = v

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(grid, cmap="magma", origin="upper")
    ax.set_xticks(range(width))
    ax.set_yticks(range(height))
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.grid(True, linewidth=0.5, alpha=0.5)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_distance_curve(
    distances: List[int | None],
    out_path: str,
    label: str,
) -> None:
    xs = list(range(len(distances)))
    ys = [d if d is not None else float("nan") for d in distances]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(xs, ys, color="black", linewidth=2, label=label)
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance to goal")
    ax.set_title("Distance-to-goal over time")
    ax.grid(True, linewidth=0.5, alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
