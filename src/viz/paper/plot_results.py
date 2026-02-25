from __future__ import annotations

import csv
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def load_csv(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(x: str) -> float:
    return float(x) if x not in ("", None) else 0.0


def to_bool(x: str) -> bool:
    return str(x).lower() == "true"


def main() -> None:
    rows = load_csv("data/aggregate_llm.csv")

    # Goal rate by model (all conditions pooled)
    by_model: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    labels = []
    goal_rates = []
    for model, items in by_model.items():
        n = len(items)
        goal = sum(1 for i in items if to_bool(i["reached_goal"]))
        labels.append(model)
        goal_rates.append(goal / n if n else 0.0)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(labels, goal_rates, color="steelblue")
    ax.set_ylabel("Goal rate")
    ax.set_title("Goal rate by model (all conditions)")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig("data/plot_goal_rate_by_model.png", dpi=150)
    plt.close(fig)

    # History sweep (per model, local_grid true, structured false)
    hist_groups: Dict[str, Dict[int, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["local_grid"] != "True" or r["structured_output"] != "False":
            continue
        model = r["model"]
        h = int(r["history_steps"])
        hist_groups[model][h].append(r)

    fig, ax = plt.subplots(figsize=(6, 3))
    for model, hmap in hist_groups.items():
        hs = sorted(hmap.keys())
        grs = []
        for h in hs:
            items = hmap[h]
            goal = sum(1 for i in items if to_bool(i["reached_goal"]))
            grs.append(goal / len(items) if items else 0.0)
        ax.plot(hs, grs, marker="o", label=model)
    ax.set_xlabel("History steps")
    ax.set_ylabel("Goal rate")
    ax.set_title("Goal rate vs history (local grid on)")
    ax.grid(True, linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("data/plot_goal_rate_by_history.png", dpi=150)
    plt.close(fig)

    # Maze difficulty comparison (simple vs hard)
    maze_groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for r in rows:
        maze = r.get("maze", "")
        if maze == "":
            # Backfill maze label from max_steps if older runs didn't log maze.
            try:
                max_steps = int(r.get("max_steps", "0"))
            except ValueError:
                max_steps = 0
            maze = "hard" if max_steps >= 80 else "simple"
        key = (r["model"], maze)
        maze_groups[key].append(r)

    labels = []
    goal_rates = []
    for (model, maze), items in sorted(maze_groups.items()):
        goal = sum(1 for i in items if to_bool(i["reached_goal"]))
        labels.append(f"{model}|{maze}")
        goal_rates.append(goal / len(items) if items else 0.0)

    if labels:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.bar(labels, goal_rates, color="darkorange")
        ax.set_ylabel("Goal rate")
        ax.set_title("Goal rate by model and maze")
        plt.xticks(rotation=30, ha="right", fontsize=7)
        fig.tight_layout()
        fig.savefig("data/plot_goal_rate_by_maze.png", dpi=150)
        plt.close(fig)
    else:
        print("No maze-labeled rows found. Re-run aggregate_runs.py after new runs.")

    # Fallback rate by model (compliance)
    labels = []
    fallback_rates = []
    for model, items in by_model.items():
        n = len(items)
        avg_fallback = sum(to_float(i["fallback_rate"]) for i in items) / n if n else 0.0
        labels.append(model)
        fallback_rates.append(avg_fallback)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(labels, fallback_rates, color="crimson")
    ax.set_ylabel("Fallback rate")
    ax.set_title("Fallback rate by model (format compliance)")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig("data/plot_fallback_rate_by_model.png", dpi=150)
    plt.close(fig)

    # Invalid rate by model
    labels = []
    invalid_rates = []
    for model, items in by_model.items():
        n = len(items)
        avg_invalid = sum(to_float(i["invalid_move_rate"]) for i in items) / n if n else 0.0
        labels.append(model)
        invalid_rates.append(avg_invalid)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(labels, invalid_rates, color="slateblue")
    ax.set_ylabel("Invalid move rate")
    ax.set_title("Invalid move rate by model")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig("data/plot_invalid_rate_by_model.png", dpi=150)
    plt.close(fig)

    # Loop rate by model
    labels = []
    loop_rates = []
    for model, items in by_model.items():
        n = len(items)
        avg_loop = sum(to_float(i["loop_rate"]) for i in items) / n if n else 0.0
        labels.append(model)
        loop_rates.append(avg_loop)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(labels, loop_rates, color="seagreen")
    ax.set_ylabel("Loop rate")
    ax.set_title("Loop rate by model")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig("data/plot_loop_rate_by_model.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
