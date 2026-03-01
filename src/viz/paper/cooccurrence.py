from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

RUNS_DIR = Path("data/runs")
OUT_CSV = Path("data/cooccurrence.csv")
OUT_PLOT = Path("data/plot_cooccurrence_overall.png")
OUT_SCATTER = Path("data/plot_fallback_vs_invalid.png")
OUT_SUMMARY = Path("data/cooccurrence_summary.txt")

THRESHOLDS = [0.1, 0.3]


@dataclass
class RunFlags:
    model: str
    history_steps: int
    local_grid: bool
    structured_output: bool
    max_steps: int
    maze: str
    has_loop: bool
    has_invalid: bool
    has_fallback: bool
    loop_rate: float
    invalid_rate: float
    fallback_rate: float


def iter_run_dirs() -> Iterable[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted([p for p in RUNS_DIR.iterdir() if p.is_dir()])


def load_config(run_dir: Path) -> dict:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return {}
    with cfg_path.open() as f:
        return json.load(f)


def read_step_log(run_dir: Path) -> list[dict]:
    log_path = run_dir / "step_log_llm.jsonl"
    if not log_path.exists():
        return []
    rows: list[dict] = []
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def flag_run(run_dir: Path) -> RunFlags | None:
    cfg = load_config(run_dir)
    rows = read_step_log(run_dir)
    if not rows:
        return None

    total = len(rows)
    loop_count = sum(1 for r in rows if r.get("loop_event", False))
    invalid_count = sum(
        1
        for r in rows
        if (not r.get("valid_move", True)) or r.get("hit_wall", False) or r.get("out_of_bounds", False)
    )
    fallback_count = sum(1 for r in rows if r.get("fallback_used", False))

    loop_rate = loop_count / total if total else 0.0
    invalid_rate = invalid_count / total if total else 0.0
    fallback_rate = fallback_count / total if total else 0.0

    has_loop = loop_count > 0
    has_invalid = invalid_count > 0
    has_fallback = fallback_count > 0

    return RunFlags(
        model=str(cfg.get("ollama_model", "unknown")),
        history_steps=int(cfg.get("history_steps", 0)),
        local_grid=bool(cfg.get("local_grid", False)),
        structured_output=bool(cfg.get("structured_output", False)),
        max_steps=int(cfg.get("max_steps", 0)),
        maze=str(cfg.get("maze", "")),
        has_loop=has_loop,
        has_invalid=has_invalid,
        has_fallback=has_fallback,
        loop_rate=loop_rate,
        invalid_rate=invalid_rate,
        fallback_rate=fallback_rate,
    )


def write_csv(flags: list[RunFlags]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "history_steps",
                "local_grid",
                "structured_output",
                "max_steps",
                "maze",
                "has_loop",
                "has_invalid",
                "has_fallback",
                "loop_rate",
                "invalid_rate",
                "fallback_rate",
            ]
        )
        for r in flags:
            writer.writerow(
                [
                    r.model,
                    r.history_steps,
                    int(r.local_grid),
                    int(r.structured_output),
                    r.max_steps,
                    r.maze,
                    int(r.has_loop),
                    int(r.has_invalid),
                    int(r.has_fallback),
                    f"{r.loop_rate:.6f}",
                    f"{r.invalid_rate:.6f}",
                    f"{r.fallback_rate:.6f}",
                ]
            )


def plot_overall(flags: list[RunFlags]) -> None:
    combos = Counter()
    for r in flags:
        key = (r.has_loop, r.has_invalid, r.has_fallback)
        combos[key] += 1

    total = sum(combos.values()) or 1
    labels = []
    values = []
    for key in sorted(combos.keys()):
        label = f"L{int(key[0])}-I{int(key[1])}-F{int(key[2])}"
        labels.append(label)
        values.append(combos[key] / total)

    plt.figure(figsize=(7, 3.5))
    plt.bar(labels, values, color="#4C78A8")
    plt.title("Failure Co-Occurrence (Overall)")
    plt.ylabel("Proportion of episodes")
    plt.xlabel("Loop / Invalid / Fallback flags")
    plt.ylim(0, 1)
    plt.tight_layout()
    OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PLOT, dpi=200)
    plt.close()


def plot_scatter(flags: list[RunFlags]) -> None:
    xs = [r.fallback_rate for r in flags]
    ys = [r.invalid_rate for r in flags]
    plt.figure(figsize=(4.5, 4))
    plt.scatter(xs, ys, s=18, alpha=0.6, color="#F58518")
    plt.title("Fallback vs Invalid Rate (per episode)")
    plt.xlabel("Fallback rate")
    plt.ylabel("Invalid-move rate")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    OUT_SCATTER.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_SCATTER, dpi=200)
    plt.close()


def corr(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n == 0:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0 or deny == 0:
        return float("nan")
    return num / (denx * deny)


def write_summary(flags: list[RunFlags]) -> None:
    xs = [r.fallback_rate for r in flags]
    ys = [r.invalid_rate for r in flags]
    zs = [r.loop_rate for r in flags]
    corr_f_i = corr(xs, ys)
    corr_f_l = corr(xs, zs)
    corr_i_l = corr(ys, zs)

    lines = []
    lines.append(f"episodes={len(flags)}")
    lines.append(f"corr(fallback, invalid)={corr_f_i:.3f}")
    lines.append(f"corr(fallback, loop)={corr_f_l:.3f}")
    lines.append(f"corr(invalid, loop)={corr_i_l:.3f}")

    for th in THRESHOLDS:
        both_fi = sum(1 for r in flags if r.fallback_rate >= th and r.invalid_rate >= th)
        both_fl = sum(1 for r in flags if r.fallback_rate >= th and r.loop_rate >= th)
        both_il = sum(1 for r in flags if r.invalid_rate >= th and r.loop_rate >= th)
        all_three = sum(
            1
            for r in flags
            if r.fallback_rate >= th and r.invalid_rate >= th and r.loop_rate >= th
        )
        lines.append(
            f"threshold>={th:.1f}: "
            f"fallback+invalid={both_fi/len(flags):.3f}, "
            f"fallback+loop={both_fl/len(flags):.3f}, "
            f"invalid+loop={both_il/len(flags):.3f}, "
            f"all_three={all_three/len(flags):.3f}"
        )

    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.write_text("\n".join(lines))


def main() -> None:
    runs: list[RunFlags] = []
    for run_dir in iter_run_dirs():
        flagged = flag_run(run_dir)
        if flagged is not None:
            runs.append(flagged)

    if not runs:
        print("No runs found.")
        return

    write_csv(runs)
    plot_overall(runs)
    plot_scatter(runs)
    write_summary(runs)
    print("Co-occurrence complete")
    print(f"Output CSV: {OUT_CSV}")
    print(f"Output plot: {OUT_PLOT}")
    print(f"Output scatter: {OUT_SCATTER}")
    print(f"Output summary: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
