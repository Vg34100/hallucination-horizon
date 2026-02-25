from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from typing import Dict, List


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    runs_dir = "data/runs"
    out_dir = "data/representative"
    os.makedirs(out_dir, exist_ok=True)

    # For each model, pick a run with reached_goal True if possible, else lowest min_distance.
    candidates: Dict[str, List[tuple]] = defaultdict(list)
    for run in os.listdir(runs_dir):
        run_path = os.path.join(runs_dir, run)
        if not os.path.isdir(run_path):
            continue
        cfg_path = os.path.join(run_path, "config.json")
        sum_path = os.path.join(run_path, "summary_llm.json")
        if not os.path.exists(cfg_path) or not os.path.exists(sum_path):
            continue
        cfg = load_json(cfg_path)
        summary = load_json(sum_path)
        model = cfg.get("ollama_model", "")
        if not model:
            continue
        reached = bool(summary.get("reached_goal", False))
        min_dist = summary.get("min_distance", 999)
        candidates[model].append((reached, min_dist, run_path))

    for model, items in candidates.items():
        # Sort: reached first, then min distance.
        items.sort(key=lambda x: (not x[0], x[1]))
        best = items[0][2]
        model_dir = os.path.join(out_dir, model.replace(":", "_"))
        os.makedirs(model_dir, exist_ok=True)
        for fname in [
            "traj_llm.png",
            "heat_invalid_llm.png",
            "heat_loops_llm.png",
            "dist_curve_llm.png",
            "summary_llm.json",
            "config.json",
        ]:
            src = os.path.join(best, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(model_dir, fname))

        print(f"Model {model}: selected {os.path.basename(best)}")


if __name__ == "__main__":
    main()
