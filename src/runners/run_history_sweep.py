from __future__ import annotations

import os
import sys

# Allow running as a script from repo root.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="History sweep runner")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--histories", type=str, default="0,5,10,20")
    parser.add_argument("--local-grid", action="store_true")
    parser.add_argument("--local-grid-radius", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    histories = [int(h.strip()) for h in args.histories.split(",") if h.strip()]
    env = os.environ.copy()
    env["OLLAMA_MODEL"] = args.model

    for h in histories:
        cmd = [
            sys.executable,
            "src/runners/run_experiment.py",
            "--max-steps",
            str(args.max_steps),
            "--runs",
            str(args.runs),
            "--seed",
            str(args.seed),
        ]
        if h:
            cmd += ["--history-steps", str(h)]
        if args.local_grid:
            cmd += ["--local-grid", "--local-grid-radius", str(args.local_grid_radius)]
        print("\nHISTORY:", h)
        subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
