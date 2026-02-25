# Project Plan: Mapping the Hallucination Horizon

## Phase 0 — Scope + Framing (1–2 days)
- Define the research question in 1–2 sentences.
- Specify course methods used: ABM + NLP/ML (LLM as agent policy).
- Decide evaluation metrics (failure taxonomy).
- Confirm model access (Ollama via Tailscale IP).
- Draft 6–8 bullet “Idea Presentation” outline.

Deliverable:
- Project idea writeup (1 page) or slide outline.

## Phase 1 — Minimal Prototype (Week 1)
- Implement grid environment + walls + goal.
- Implement A* baseline.
- Implement Random + Greedy baselines.
- Add logging (JSONL).
- Create initial plots: trajectory overlay + invalid-move heatmap.

Deliverable:
- 1 run folder with logs + 2 plots.

## Phase 2 — LLM Integration (Week 2)
- Add Ollama provider.
- Add strict output parsing.
- Add “reasoning text” logging.
- Run a tiny maze demo with LLM agent.

Deliverable:
- LLM run logs + visualizations.

## Phase 3 — Diagnostic Taxonomy (Week 2–3)
- Define and implement failure modes:
  - Loop Rate
  - Reality Gap (invalid moves)
  - Goal Drift
- Add per-step flags + aggregate rates.
- Create diagnostic visuals:
  - Heatmap of failure origins
  - Divergence plot vs A*

Deliverable:
- 2–3 diagnostic plots for one condition.

## Phase 4 — Experimental Suite (Week 3–4)
- Define 2–3 environment complexities.
- Run 20–30 seeds per condition.
- Summarize metrics across runs.

Deliverable:
- Summary table + plots for paper/presentation.

## Phase 5 — Communication Assets (Week 4–5)
- Slides: Problem / Methods / Findings.
- 5–10 min video.
- Final writeup with “AI use” transparency section.

Deliverable:
- Presentation + final paper draft.

## Notes
- `src/viz/paper/` contains plots and selection helpers for paper figures.
