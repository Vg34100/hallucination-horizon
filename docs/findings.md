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
