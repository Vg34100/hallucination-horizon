from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

RUNS_DIR = Path("data/runs")
OUT_CSV = Path("data/context_length.csv")
OUT_SUMMARY = Path("data/context_length_summary.txt")


def iter_run_dirs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted([p for p in RUNS_DIR.iterdir() if p.is_dir()])


def load_config(run_dir: Path) -> dict:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return {}
    with cfg_path.open() as f:
        return json.load(f)


def read_prompts(run_dir: Path) -> list[str]:
    log_path = run_dir / "step_log_llm.jsonl"
    if not log_path.exists():
        return []
    prompts: list[str] = []
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = row.get("observation_text", "")
            if prompt:
                prompts.append(prompt)
    return prompts


def summarize_prompts(prompts: list[str]) -> dict[str, float]:
    if not prompts:
        return {
            "chars_mean": 0.0,
            "chars_max": 0.0,
            "words_mean": 0.0,
            "words_max": 0.0,
        }
    char_lengths = [len(p) for p in prompts]
    word_lengths = [len(p.split()) for p in prompts]
    return {
        "chars_mean": sum(char_lengths) / len(char_lengths),
        "chars_max": max(char_lengths),
        "words_mean": sum(word_lengths) / len(word_lengths),
        "words_max": max(word_lengths),
    }


def main() -> None:
    rows = []
    by_history: dict[int, list[dict[str, float]]] = defaultdict(list)

    for run_dir in iter_run_dirs():
        cfg = load_config(run_dir)
        prompts = read_prompts(run_dir)
        if not prompts:
            continue
        stats = summarize_prompts(prompts)
        history_steps = int(cfg.get("history_steps", 0))
        rows.append(
            {
                "run_dir": run_dir.name,
                "model": cfg.get("ollama_model", "unknown"),
                "history_steps": history_steps,
                "local_grid": int(bool(cfg.get("local_grid", False))),
                "structured_output": int(bool(cfg.get("structured_output", False))),
                "max_steps": int(cfg.get("max_steps", 0)),
                "maze": cfg.get("maze", ""),
                **stats,
            }
        )
        by_history[history_steps].append(stats)

    if not rows:
        print("No prompts found.")
        return

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_dir",
                "model",
                "history_steps",
                "local_grid",
                "structured_output",
                "max_steps",
                "maze",
                "chars_mean",
                "chars_max",
                "words_mean",
                "words_max",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    lines = ["Context length summary (by history_steps):"]
    for h in sorted(by_history.keys()):
        stats = by_history[h]
        chars_max = max(s["chars_max"] for s in stats)
        words_max = max(s["words_max"] for s in stats)
        chars_mean = sum(s["chars_mean"] for s in stats) / len(stats)
        words_mean = sum(s["words_mean"] for s in stats) / len(stats)
        lines.append(
            f"- history={h}: chars_max={chars_max:.0f}, words_max={words_max:.0f}, "
            f"chars_mean={chars_mean:.0f}, words_mean={words_mean:.0f}"
        )

    OUT_SUMMARY.write_text("\n".join(lines))
    print("Context-length analysis complete")
    print(f"Output CSV: {OUT_CSV}")
    print(f"Output summary: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
