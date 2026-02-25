from __future__ import annotations

import argparse
import importlib
import os
import sys

# Allow running as a script from repo root.
sys.path.append(os.path.dirname(__file__))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project runner")
    parser.add_argument(
        "--mode",
        type=str,
        default="experiment",
        choices=["experiment", "batch", "sweep", "aggregate", "viewer", "viewer-mpl", "full-suite", "plot", "pick"],
    )
    parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def run_module(module_path: str, argv: list[str]) -> None:
    if argv and argv[0] == "--":
        argv = argv[1:]
    sys.argv = [module_path] + argv
    mod = importlib.import_module(module_path)
    if hasattr(mod, "main"):
        mod.main()


def main() -> None:
    args = parse_args()
    if args.mode == "experiment":
        run_module("runners.run_experiment", args.args)
    elif args.mode == "batch":
        run_module("runners.run_batch", args.args)
    elif args.mode == "sweep":
        run_module("runners.run_history_sweep", args.args)
    elif args.mode == "aggregate":
        run_module("runners.aggregate_runs", args.args)
    elif args.mode == "full-suite":
        run_module("runners.run_full_suite", args.args)
    elif args.mode == "plot":
        run_module("viz.paper.plot_results", args.args)
    elif args.mode == "pick":
        run_module("viz.paper.pick_representative", args.args)
    elif args.mode == "viewer":
        run_module("viz.tk_viewer", args.args)
    elif args.mode == "viewer-mpl":
        run_module("viz.mpl_viewer", args.args)


if __name__ == "__main__":
    main()
