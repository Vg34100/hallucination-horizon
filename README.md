# LLM Spatial Failure Modes (Gridworld)

This is my CSS final project codebase. The idea is simple: build a gridworld where an LLM controls an agent, compare it to A* (ground truth), and log where the LLM fails (invalid moves, loops, goal drift). I care less about a single success rate and more about *diagnosing failure modes* and visualizing them.

## What’s here
- Gridworld environment + A* planner
- Baselines: Random, Greedy, A*
- LLM agent via Ollama (local)
- Logging + metrics + plots
- Batch runs + aggregation
- Interactive Tk viewer

## Quick start
```bash
# single run
python3 src/main.py --mode experiment

# interactive viewer
python3 src/main.py --mode viewer

# batch runs (example)
python3 src/main.py --mode batch -- --models llama3.2:3b --history-steps 10 --local-grid --runs 5 --seed 1

# aggregate results
python3 src/main.py --mode aggregate
```

## Common flags (experiment)
```bash
python3 src/main.py --mode experiment -- --history-steps 10 --local-grid
python3 src/main.py --mode experiment -- --dynamic-walls --flip-every 5
python3 src/main.py --mode experiment -- --maze hard
```

## Notes
- I default the Ollama base URL and model in code (see `src/runners/run_experiment.py`).
- The Tk viewer lets me pick a model from my running Ollama instance.
- If the model doesn’t emit a valid direction, the agent falls back to a random valid move (this is logged).

## Repo layout
```
src/
  core/          # env, planner, metrics
  agents/        # baselines + LLM agent + Ollama client
  prompts/       # prompt builders
  viz/           # plots + viewers
  runners/       # run scripts
  main.py        # entry point
```

> Paper title: “Mapping the Hallucination Horizon: Visual Diagnostics of LLM Spatial Failure Modes”
