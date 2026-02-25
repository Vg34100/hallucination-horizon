# Slide 1 — Title
**Title:** Mapping the Hallucination Horizon: Visual Diagnostics of LLM Spatial Failure Modes
**Subtitle:** ABM Gridworld + LLM Agents
**Name/affiliation:** Pablo Rodriguez, ViDi Lab / CSS

**Speaker notes (verbatim):**
Today I’m presenting a diagnostic study of spatial failure modes in LLM agents. The core idea is simple: LLMs do well on static benchmarks, but their step‑by‑step behavior in an environment is still poorly understood. I build a gridworld ABM to visualize exactly how and where they fail.

---

# Slide 2 — Problem
**Bullet points:**
- LLMs pass static spatial tests
- But interactive navigation requires state tracking
- We lack diagnostic tools for *how* failures happen

**Speaker notes (verbatim):**
The problem is that most spatial benchmarks are static question answering. But navigation is interactive and requires tracking state across steps. We need tools that show not just whether an agent succeeds, but how it fails.

---

# Slide 3 — Method (Environment + Agents)
**Bullet points:**
- 2D grid with walls, start, goal
- Actions: N, S, E, W
- Baselines: A*, Random, Greedy
- LLM agent gets text observations

**Speaker notes (verbatim):**
I use a gridworld with walls, a start, and a goal. The agent moves one step at a time. I compare an optimal A* path to random and greedy baselines, and then to an LLM agent that only sees text observations.

---

# Slide 4 — Failure Diagnostics
**Bullet points:**
- Loop rate = memory failure
- Reality gap = invalid moves + fallback
- Goal drift = distance‑to‑goal curve

**Speaker notes (verbatim):**
I diagnose three failure modes. Loop rate captures memory failure and cycling. Reality gap measures invalid moves and fallback actions when the model fails to output a usable direction. Goal drift tracks whether the agent gets closer to the goal over time compared to optimal paths.

---

# Slide 5 — Results: Model Comparison
**Visual:** Goal rate by model (plot)

**Speaker notes (verbatim):**
Across models, gpt‑oss:20b has the highest goal rate under several conditions, but still shows high invalid‑move and fallback rates. Llama3.2:3b is competitive, while phi3 and qwen perform worse, especially on compliance with output constraints.

---


---

# Slide 6 — Interactive Viewer
**Bullet points:**
- Live grid rendering
- Prompt + raw output panel
- Step/run controls + history/grid tuning

**Speaker notes (verbatim):**
I also built a lightweight interactive viewer. It renders the grid, shows the prompt and raw LLM output, and lets me step through decisions or run continuously while changing history length and local-grid settings. This made it easy to observe failure modes in real time.
# Slide 6 — Results: History & Maze Difficulty
**Visuals:** Goal rate vs history plot + Goal rate by maze plot

**Speaker notes (verbatim):**
History length does not monotonically improve performance. In fact, history 10 often performs better than history 20, suggesting longer context can add noise. Hard mazes reduce goal rates across all models, and dynamic walls strongly degrade performance.

---

# Slide 7 — Diagnostic Visuals
**Visuals:**
- Trajectory overlay
- Invalid‑move heatmap
- Distance‑to‑goal curve

**Speaker notes (verbatim):**
These visual diagnostics show the difference between success and failure. Heatmaps highlight where invalid moves cluster, trajectories reveal loops, and distance‑to‑goal curves show optimization failure even when moves are valid.

---

# Slide 8 — Takeaways
**Bullet points:**
- LLM navigation is unstable
- Success depends on observation design
- Diagnostics reveal failure structure

**Speaker notes (verbatim):**
The main takeaway is that LLM navigation is fragile and highly sensitive to observation design. Diagnostic visuals reveal structured failure modes that success rates alone hide. This supports the need for interactive evaluation when using LLMs as agents.
