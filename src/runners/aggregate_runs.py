from __future__ import annotations

import os
import sys

# Allow running as a script from repo root.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import csv
import glob
import json
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    runs = sorted(glob.glob("data/runs/run_*"))
    rows: List[dict] = []

    for run_dir in runs:
        config_path = os.path.join(run_dir, "config.json")
        summary_path = os.path.join(run_dir, "summary_llm.json")
        if not os.path.exists(config_path) or not os.path.exists(summary_path):
            continue
        config = load_json(config_path)
        summary = load_json(summary_path)

        row = {
            "run_id": os.path.basename(run_dir),
            "model": config.get("ollama_model", ""),
            "history_steps": config.get("history_steps", 0),
            "local_grid": config.get("local_grid", False),
            "structured_output": config.get("structured_output", False),
            "max_steps": config.get("max_steps", 0),
            "reached_goal": summary.get("reached_goal", False),
            "final_distance": summary.get("final_distance", None),
            "min_distance": summary.get("min_distance", None),
            "invalid_move_rate": summary.get("invalid_move_rate", 0.0),
            "loop_rate": summary.get("loop_rate", 0.0),
            "fallback_rate": summary.get("fallback_rate", 0.0),
        }
        rows.append(row)

    out_path = "data/aggregate_llm.csv"
    os.makedirs("data", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    # Aggregate by condition
    grouped: Dict[Tuple, List[dict]] = defaultdict(list)
    for r in rows:
        key = (
            r["model"],
            r["history_steps"],
            r["local_grid"],
            r["structured_output"],
            r["max_steps"],
        )
        grouped[key].append(r)

    print("Aggregate summary by condition:")
    labels = []
    goal_rates = []
    history_points = []
    history_goal_rates = []
    for key, items in grouped.items():
        model, history_steps, local_grid, structured_output, max_steps = key
        n = len(items)
        goal_rate = sum(1 for i in items if i["reached_goal"]) / n if n else 0.0
        avg_min_dist = sum(i["min_distance"] for i in items if i["min_distance"] is not None) / n
        avg_invalid = sum(i["invalid_move_rate"] for i in items) / n
        avg_fallback = sum(i["fallback_rate"] for i in items) / n
        print(
            f"- model={model} history={history_steps} local_grid={local_grid} "
            f"structured={structured_output} max_steps={max_steps} "
            f"runs={n} goal_rate={goal_rate:.2f} "
            f"avg_min_dist={avg_min_dist:.2f} avg_invalid={avg_invalid:.2f} "
            f"avg_fallback={avg_fallback:.2f}"
        )
        label = f"{model}|h{history_steps}|g{int(local_grid)}|s{int(structured_output)}"
        labels.append(label)
        goal_rates.append(goal_rate)
        history_points.append((model, history_steps, local_grid, structured_output, max_steps, goal_rate))

    if labels:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar(range(len(labels)), goal_rates, color="steelblue")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Goal rate")
        ax.set_title("Goal rate by condition")
        fig.tight_layout()
        fig.savefig("data/aggregate_goal_rate.png", dpi=150)
        plt.close(fig)

    # History sweep plot (group by model + local_grid + structured)
    groups: Dict[Tuple, List[Tuple[int, float]]] = defaultdict(list)
    for model, h, lg, s, max_steps, gr in history_points:
        key = (model, lg, s, max_steps)
        groups[key].append((h, gr))

    if groups:
        fig, ax = plt.subplots(figsize=(5, 3))
        for (model, lg, s, max_steps), items in groups.items():
            items.sort(key=lambda x: x[0])
            hs = [i[0] for i in items]
            grs = [i[1] for i in items]
            label = f"{model}|grid={int(lg)}|struct={int(s)}"
            ax.plot(hs, grs, marker="o", label=label)
        ax.set_xlabel("History steps")
        ax.set_ylabel("Goal rate")
        ax.set_title("Goal rate vs history")
        ax.grid(True, linewidth=0.5, alpha=0.5)
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig("data/aggregate_goal_rate_by_history.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
