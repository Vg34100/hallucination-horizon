from __future__ import annotations

import itertools
import os
import subprocess
import sys


def run_case(history_steps: int, local_grid: bool, structured: bool) -> None:
    cmd = [sys.executable, "src/run_experiment.py", "--max-steps", "60"]
    if history_steps:
        cmd += ["--history-steps", str(history_steps)]
    if local_grid:
        cmd += ["--local-grid"]
    if structured:
        cmd += ["--structured-output"]

    print("\nCASE:", "history=", history_steps, "local_grid=", local_grid, "structured=", structured)
    subprocess.run(cmd, check=True)


def main() -> None:
    histories = [0, 10]
    local_grids = [False, True]
    structured = [False, True]

    for h, lg, s in itertools.product(histories, local_grids, structured):
        run_case(h, lg, s)


if __name__ == "__main__":
    main()
