# Findings Log

## 2026-02-28
- Co-occurrence analysis (per-episode rates + thresholds)
  - Purpose: The original co-occurrence plot was saturated because almost any episode with a single fallback/invalid counted as “has_* = 1.” We now compute per-episode rates and report correlations plus thresholded overlaps to better understand how failures co-occur.
  - Commit: `feat: add co-occurrence analysis for failure modes`
  - Commands:
    - `python3 src/main.py --mode cooccurrence`
  - Outputs:
    - `data/cooccurrence.csv`
    - `data/plot_cooccurrence_overall.png`
    - `data/plot_fallback_vs_invalid.png`
    - `data/cooccurrence_summary.txt`
  - Findings:
    - corr(fallback, invalid) = -0.808 (fallback appears to suppress invalid-rate counts)
    - corr(fallback, loop) = 0.547 (fallback co-occurs with looping behavior)
    - corr(invalid, loop) = -0.694 (episodes with high invalid rates tend to have lower loop rates)
    - At threshold >=0.1, co-occurrence is common (fallback+invalid 75.7%, all three 53.0%), but at >=0.3, high-rate overlaps vanish.

- Context length audit (prompt sizes vs history)
  - Purpose: verify prompts are far below the 64k context budget and rule out truncation as a primary cause of failures.
  - Commit: `chore: add context-length audit`
  - Commands:
    - `python3 src/main.py --mode context-length`
  - Outputs:
    - `data/context_length.csv`
    - `data/context_length_summary.txt`
  - Findings:
    - history=0: chars_max=11147, words_max=2213 (largest prompts observed)
    - history=20: chars_max=3978, words_max=770
    - All observed prompts are well below a 64k context window, so truncation is unlikely to explain failures.

- Planning-prompt pilot (gpt-oss:20b, simple maze, history=10, local grid)
  - Purpose: test whether a short explicit planning step changes loop/invalid behavior without re-running the full suite.
  - Commit: `feat: add planning-prompt variant for LLM`
  - Commands:
    - `python3 src/main.py --mode experiment -- --maze simple --history-steps 10 --local-grid --plan-prompt --runs 5`
  - Outputs:
    - `data/runs/run_20260301_115608_1` through `_5`
  - Findings (LLM only):
    - goal_rate=0.80 (4/5)
    - avg_invalid_move_rate=0.384
    - avg_loop_rate=0.111
    - avg_fallback_rate=0.293
