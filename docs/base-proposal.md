## Goal of the first tiny test

Make a grid maze where:

* A* solves it (ground truth).
* An “LLM agent” gets a text view each step and chooses an action.
* You log failures: wall hits, loops, goal drift.
* You output 2 plots: trajectory overlay and a heatmap of failures.

You can do this with zero ABM framework first. Then, if needed, wrap it into Mesa after.

---

## Tools and libraries

Core:

* Python 3.11+
* `numpy`
* `matplotlib`
* `networkx` (easy A* and graph utilities)
* `pydantic` (optional, for clean typed logs)
* `tqdm` (optional progress bars)

LLM:

* `openai` (if using GPT models)
* or `ollama` (local Llama 3) via HTTP
* or `anthropic` (Claude) if you go that route

Nice-to-have:

* `pygame` (only if you want live animation; not required)
* `mesa` (later, if you want to call it ABM in the code structure, but you already have ABM conceptually)

I would start without Mesa. A gridworld with an agent is still ABM logic.

---

## Folder layout

Keep it simple and separated.

```
project/
  README.md
  requirements.txt
  config.yaml
  src/
    env_grid.py
    planner_astar.py
    agent_llm.py
    agent_baseline.py
    prompts.py
    metrics.py
    run_experiment.py
    viz.py
  data/
    runs/
      run_YYYYMMDD_HHMMSS/
        config.json
        step_log.jsonl
        summary.json
        traj.png
        heat_invalid.png
        heat_loops.png
```

---

## Step-by-step build plan

### Step 1: Grid environment (no LLM yet)

Implement a minimal environment class:

* state:

  * width, height
  * walls set
  * start, goal
  * agent position
* actions:

  * `N, S, E, W`
* step function:

  * if move hits wall or outside: position unchanged, mark invalid
  * else move

Dynamic twist (optional in first test):

* every `k` steps, flip one wall cell (add/remove) from a allowed list

Deliverable:

* a function to generate a fixed maze (hardcode one) and a random maze generator later

Test:

* unit test: moving into wall does not change position

### Step 2: A* control agent (ground truth)

Use `networkx.astar_path` or your own A*.

* Build graph nodes for open cells
* Compute shortest path start -> goal

Deliverable:

* path list of coordinates
* expected steps length

Test:

* ensure path exists
* ensure path never crosses walls

### Step 3: LLM agent interface (stub first)

Before hitting any API, create an agent that returns moves.

Start with:

* random agent
* greedy agent that moves closer to goal if possible

This makes your pipeline testable without LLM cost.

Deliverable:

* `agent.choose_action(observation_text) -> str`

Test:

* run 10 episodes quickly and save logs and plots

### Step 4: Observation text format

Define what the agent sees each step. Keep it small and consistent:

Example:

* current (x, y)
* goal (gx, gy)
* 4-neighborhood blocked/open
* last action
* step count

Observation string:

```
You are at (3,5). Goal is at (9,1).
North: wall. South: open. East: open. West: wall.
Last move: East.
Choose one move from: N,S,E,W.
Return only the letter.
```

Important: constrain output hard.

### Step 5: Plug in a real LLM call (single step)

Add a provider layer:

* `OpenAIProvider`
* `OllamaProvider`
* `AnthropicProvider`

Each takes `(prompt) -> text`.

Then parse the output:

* normalize
* if invalid, treat as “no-op” and count a parse error

Safety for experiments:

* set max tokens very low (like 5 to 20)
* use temperature 0 for repeatability

Test:

* one short run on a tiny maze (5x5) for 20 steps max

### Step 6: Logging (this matters for your paper-like results)

Log each step as JSON lines.

Fields:

* run_id
* step
* pos_before, pos_after
* action_raw, action_parsed
* valid_move (bool)
* hit_wall (bool)
* distance_to_goal_before/after (Manhattan)
* loop_detected (bool)
* walls_changed (bool, if dynamic)
* observation_text (optional, can store but file gets big)
* model_name

Deliverable:

* `step_log.jsonl`
* `summary.json` (counts and rates)

Test:

* confirm log length equals number of steps taken

### Step 7: Metrics

Implement the three metrics you wrote:

1. Loop Rate

* Keep a dictionary of visited positions with counts
* Loop event when a position count exceeds a threshold (like 3), or when you detect a repeated sequence in last 10 steps

2. Reality Gap

* In this setup, “reality gap” can mean:

  * attempted move into wall OR out of bounds
  * also prompt-level contradiction if you include a “claim” line, but that’s optional
    So for first version: use invalid move rate as the clean proxy.

3. Goal Drift

* Compare to A* optimal distance:

  * compute shortest path length from current to goal (precompute all-pairs shortest lengths on static maze)
  * drift when distance increases or stays flat too long

Deliverable:

* per-step flags + aggregate rates

Test:

* on greedy agent, drift should be lower than random

### Step 8: Visual outputs

Create 2 to 4 plots:

1. Trajectory overlay

* maze cells as background
* A* path in one style
* LLM path in another
* start/goal markers

2. Heatmap of invalid moves

* per-cell count of invalid attempts originating from that cell

3. Heatmap of loop intensity

* per-cell visit count

4. Optional: “drift curve”

* step on x axis, distance-to-goal on y axis for LLM vs A*

Deliverable:

* PNG files saved in run folder

Test:

* confirm plots render and save for one run

---

## Minimal test you should run first (tiny and cheap)

Maze: 7x7 with a simple corridor and one dead end.

Run:

* max_steps = 40
* 3 agents:

  * A* (control)
  * random baseline
  * LLM agent (one model)

Output:

* step log
* summary counts
* trajectory plot
* invalid heatmap

Success criteria:

* pipeline runs end-to-end
* LLM produces valid action outputs most of the time
* you see at least some invalid moves or loops sometimes (if not, tweak maze difficulty)

---


```text
Implement a Python gridworld experiment with:
1) A GridEnv class (width,height,walls,start,goal,pos) with step(action) for actions N,S,E,W. Hitting a wall/out-of-bounds should keep pos unchanged and mark invalid.
2) A* planner using networkx to compute an optimal path on the static maze.
3) Agents:
   - RandomAgent
   - GreedyAgent (moves that reduce Manhattan distance if possible)
   - LLMAgent with a provider interface; for now include a MockProvider that returns random valid letters; structure code so OpenAI/Ollama providers can be added later.
4) Observation text each step: position, goal, N/S/E/W blocked/open, allowed actions; instruct agent to return only one letter.
5) Logging: write JSONL per step with pos_before/after, action_raw/parsed, invalid_move flag, distance_before/after, loop flag, model name.
6) Metrics: invalid move rate, loop rate (revisiting same cell >=3), goal drift (distance increases).
7) Visualization: save trajectory overlay (maze + A* path + agent path) and heatmap of invalid moves per cell (origin cell).
8) Provide run_experiment.py to run one maze for 3 agents and save outputs under data/runs/run_TIMESTAMP/.

Keep functions short and clear and avoid fancy patterns. Include requirements.txt.
```
