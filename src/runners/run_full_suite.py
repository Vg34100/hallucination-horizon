from __future__ import annotations

import argparse
import os
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full experiment suite runner")
    parser.add_argument("--models", type=str, default="llama3.2:3b")
    parser.add_argument("--histories", type=str, default="0,5,10,20")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--max-steps-hard", type=int, default=80)
    parser.add_argument("--dynamic-flip", type=int, default=5)
    return parser.parse_args()


def run_cmd(cmd: list[str], env: dict) -> None:
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    histories = [h.strip() for h in args.histories.split(",") if h.strip()]

    for model in models:
        env = os.environ.copy()
        env["OLLAMA_MODEL"] = model

        # Simple maze sweep
        cmd_simple = [
            sys.executable,
            "src/runners/run_history_sweep.py",
            "--model",
            model,
            "--histories",
            ",".join(histories),
            "--local-grid",
            "--runs",
            str(args.runs),
            "--seed",
            str(args.seed),
            "--max-steps",
            str(args.max_steps),
        ]
        print("\nSIMPLE MAZE:", model)
        run_cmd(cmd_simple, env)

        # Hard maze sweep
        cmd_hard = [
            sys.executable,
            "src/runners/run_history_sweep.py",
            "--model",
            model,
            "--histories",
            ",".join(histories),
            "--local-grid",
            "--runs",
            str(args.runs),
            "--seed",
            str(args.seed),
            "--max-steps",
            str(args.max_steps_hard),
        ]
        cmd_hard += ["--maze", "hard"]
        print("\nHARD MAZE:", model)
        run_cmd(cmd_hard, env)

        # Dynamic walls stress test (single history)
        cmd_dyn = [
            sys.executable,
            "src/runners/run_experiment.py",
            "--maze",
            "hard",
            "--history-steps",
            "10",
            "--local-grid",
            "--runs",
            str(args.runs),
            "--seed",
            str(args.seed),
            "--max-steps",
            str(args.max_steps_hard),
            "--dynamic-walls",
            "--flip-every",
            str(args.dynamic_flip),
        ]
        print("\nDYNAMIC WALLS:", model)
        run_cmd(cmd_dyn, env)

    # Aggregate at the end
    run_cmd([sys.executable, "src/runners/aggregate_runs.py"], os.environ.copy())


if __name__ == "__main__":
    main()
