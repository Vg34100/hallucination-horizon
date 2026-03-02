from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

IN_CSV = Path("data/aggregate_llm.csv")
OUT_TXT = Path("data/stats_summary.txt")


def mean_ci(series: pd.Series) -> tuple[float, float]:
    n = series.count()
    if n == 0:
        return float("nan"), float("nan")
    mean = series.mean()
    std = series.std(ddof=1) if n > 1 else 0.0
    # 95% CI using normal approximation
    ci = 1.96 * (std / math.sqrt(n)) if n > 1 else 0.0
    return mean, ci


def format_line(label: str, mean: float, ci: float) -> str:
    return f"{label}: mean={mean:.3f}, 95%CI=±{ci:.3f}"


def main() -> None:
    if not IN_CSV.exists():
        print(f"Missing {IN_CSV}")
        return

    df = pd.read_csv(IN_CSV)

    lines = []
    lines.append("Stats summary (per-run, descriptive):")

    by_model = df.groupby("model")
    for model, sub in by_model:
        lines.append(f"\nModel: {model} (n={len(sub)})")
        for col, label in [
            ("reached_goal", "goal_rate"),
            ("invalid_move_rate", "invalid_rate"),
            ("loop_rate", "loop_rate"),
            ("fallback_rate", "fallback_rate"),
            ("min_distance", "min_distance"),
        ]:
            mean, ci = mean_ci(sub[col])
            lines.append(format_line(label, mean, ci))

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(lines))
    print("Stats summary complete")
    print(f"Output: {OUT_TXT}")


if __name__ == "__main__":
    main()
