from __future__ import annotations

import os
import sys

# Allow running as a script from repo root.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch experiment runner")
    parser.add_argument("--models", type=str, default="")
    parser.add_argument("--history-steps", type=int, default=0)
    parser.add_argument("--local-grid", action="store_true")
    parser.add_argument("--local-grid-radius", type=int, default=2)
    parser.add_argument("--structured-output", action="store_true")
    parser.add_argument("--dynamic-walls", action="store_true")
    parser.add_argument("--flip-every", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        models = [os.environ.get("OLLAMA_MODEL", "")]

    for model in models:
        if not model:
            continue
        env = os.environ.copy()
        env["OLLAMA_MODEL"] = model

        cmd = [
            sys.executable,
            "src/runners/run_experiment.py",
            "--max-steps",
            str(args.max_steps),
            "--runs",
            str(args.runs),
        ]
        if args.history_steps:
            cmd += ["--history-steps", str(args.history_steps)]
        if args.local_grid:
            cmd += ["--local-grid", "--local-grid-radius", str(args.local_grid_radius)]
        if args.structured_output:
            cmd += ["--structured-output"]
        if args.dynamic_walls:
            cmd += ["--dynamic-walls", "--flip-every", str(args.flip_every)]
        if args.seed is not None:
            cmd += ["--seed", str(args.seed)]

        print("\nMODEL:", model)
        subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
