from __future__ import annotations

import os
import sys

# Allow running as a script from repo root.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import random
from datetime import datetime
from typing import Dict, List, Tuple

from agents.astar_agent import AStarAgent
from agents.greedy_agent import GreedyAgent
from agents.observation import Observation
from agents.random_agent import RandomAgent
from agents.llm_agent import LLMAgent, OllamaProvider
from core.env_grid import GridEnv
from core.mazes import make_hard_maze, make_simple_maze
from core.metrics import MetricsState
from core.planner_astar import all_pairs_shortest_lengths, astar_path
from prompts.prompts import build_prompt, format_local_grid, format_observation
from viz.plots import plot_distance_curve, plot_heatmap, plot_trajectory

Coord = Tuple[int, int]


def run_agent(
    env: GridEnv,
    agent_name: str,
    choose_action_fn,
    run_dir: str,
    max_steps: int = 60,
    history_steps: int = 0,
    include_local_grid: bool = False,
    local_grid_radius: int = 2,
    dynamic_walls: bool = False,
    flip_every: int = 0,
    progress_every: int = 10,
) -> dict:
    # Runs a single agent for one episode and writes logs/plots.
    shortest = all_pairs_shortest_lengths(env.width, env.height, env.walls)
    astar = astar_path(env.width, env.height, env.walls, env.start, env.goal)

    metrics = MetricsState()
    invalid_origin_counts: Dict[Coord, int] = {}
    loop_counts: Dict[Coord, int] = {}
    fallback_count = 0

    path: List[Coord] = [env.pos]
    last_action = None
    history: List[str] = []
    reached_goal = False
    min_dist = None

    log_path = os.path.join(run_dir, f"step_log_{agent_name}.jsonl")
    distance_curve: List[int | None] = []
    with open(log_path, "w", encoding="utf-8") as f:
        for step in range(max_steps):
            walls_changed = False
            flipped_cell = None
            if dynamic_walls and flip_every > 0 and step > 0 and step % flip_every == 0:
                flipped_cell = env.flip_wall()
                walls_changed = flipped_cell is not None

            open_map = env.neighbors_open(env.pos)
            obs = Observation(
                pos=env.pos,
                goal=env.goal,
                open_map=open_map,
                last_action=last_action,
                step=step,
            )

            obs_text = format_observation(
                obs.pos, obs.goal, obs.open_map, obs.last_action, obs.step
            )
            local_grid = None
            if include_local_grid:
                local_grid = format_local_grid(
                    obs.pos,
                    obs.goal,
                    env.walls,
                    env.width,
                    env.height,
                    local_grid_radius,
                )
            prompt = build_prompt(obs_text, history[-history_steps:], local_grid)
            action_response = choose_action_fn(obs, prompt)
            action_raw = action_response
            action_parsed = action_response
            fallback_used = False
            if hasattr(action_response, "raw") and hasattr(action_response, "parsed"):
                action_raw = action_response.raw
                action_parsed = action_response.parsed or ""
                fallback_used = getattr(action_response, "fallback_used", False)

            action = action_parsed
            result = env.step(action)

            metrics.steps += 1
            loop_event = metrics.update_visit(result.pos_after)
            metrics.update_invalid(not result.valid_move)

            if not result.valid_move:
                invalid_origin_counts[result.pos_before] = (
                    invalid_origin_counts.get(result.pos_before, 0) + 1
                )
            if loop_event:
                loop_counts[result.pos_after] = loop_counts.get(result.pos_after, 0) + 1
            if fallback_used:
                fallback_count += 1

            dist_before = shortest.get(result.pos_before, {}).get(env.goal, None)
            dist_after = shortest.get(result.pos_after, {}).get(env.goal, None)
            goal_drift = False
            if dist_before is not None and dist_after is not None:
                goal_drift = dist_after > dist_before
                if min_dist is None or dist_after < min_dist:
                    min_dist = dist_after

            record = {
                "agent": agent_name,
                "step": step,
                "pos_before": result.pos_before,
                "pos_after": result.pos_after,
                "action_raw": action_raw,
                "action_parsed": action_parsed,
                "fallback_used": fallback_used,
                "valid_move": result.valid_move,
                "hit_wall": result.hit_wall,
                "out_of_bounds": result.out_of_bounds,
                "distance_to_goal_before": dist_before,
                "distance_to_goal_after": dist_after,
                "goal_drift": goal_drift,
                "loop_event": loop_event,
                "walls_changed": walls_changed,
                "flipped_cell": flipped_cell,
                "open_map": open_map,
                "observation_text": prompt,
            }
            f.write(json.dumps(record) + "\n")
            distance_curve.append(dist_after)

            path.append(result.pos_after)
            last_action = action
            history.append(f"Obs: {obs_text} | Action: {action}")

            if result.pos_after == env.goal:
                reached_goal = True
                break

            if progress_every and (step + 1) % progress_every == 0:
                print(f"[{agent_name}] step {step+1}/{max_steps} pos={result.pos_after}")

    summary = {
        "agent": agent_name,
        "steps": metrics.steps,
        "invalid_move_rate": metrics.rate_invalid(),
        "loop_rate": metrics.rate_loops(),
        "fallback_rate": fallback_count / metrics.steps if metrics.steps else 0.0,
        "reached_goal": reached_goal,
        "final_distance": dist_after,
        "min_distance": min_dist if min_dist is not None else dist_after,
    }

    with open(
        os.path.join(run_dir, f"summary_{agent_name}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    plot_trajectory(
        env.width,
        env.height,
        env.walls,
        env.start,
        env.goal,
        astar,
        path,
        os.path.join(run_dir, f"traj_{agent_name}.png"),
    )

    plot_heatmap(
        env.width,
        env.height,
        invalid_origin_counts,
        "Invalid Move Origins",
        os.path.join(run_dir, f"heat_invalid_{agent_name}.png"),
    )

    plot_heatmap(
        env.width,
        env.height,
        loop_counts,
        "Loop Events",
        os.path.join(run_dir, f"heat_loops_{agent_name}.png"),
    )
    plot_distance_curve(
        distance_curve,
        os.path.join(run_dir, f"dist_curve_{agent_name}.png"),
        agent_name,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gridworld LLM diagnostics")
    parser.add_argument("--maze", type=str, default="simple", choices=["simple", "hard"])
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--history-steps", type=int, default=0)
    parser.add_argument("--local-grid", action="store_true")
    parser.add_argument("--local-grid-radius", type=int, default=2)
    parser.add_argument("--structured-output", action="store_true")
    parser.add_argument("--llm-timeout", type=int, default=60)
    parser.add_argument("--llm-retries", type=int, default=1)
    parser.add_argument("--dynamic-walls", action="store_true")
    parser.add_argument("--flip-every", type=int, default=0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Pick maze and build base config.
    if args.maze == "hard":
        width, height, walls, start, goal = make_hard_maze()
    else:
        width, height, walls, start, goal = make_simple_maze()
    env = GridEnv(width, height, walls, start, goal)

    base_run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    config = {
        "maze": args.maze,
        "width": width,
        "height": height,
        "start": start,
        "goal": goal,
        "walls": walls,
        "max_steps": args.max_steps,
        "progress_every": args.progress_every,
        "history_steps": args.history_steps,
        "local_grid": args.local_grid,
        "local_grid_radius": args.local_grid_radius,
        "structured_output": args.structured_output,
        "llm_timeout": args.llm_timeout,
        "llm_retries": args.llm_retries,
        "dynamic_walls": args.dynamic_walls,
        "flip_every": args.flip_every,
        "runs": args.runs,
        "seed": args.seed,
        "ollama_base_url": os.environ.get("OLLAMA_BASE_URL", "http://100.121.2.67:11434"),
        "ollama_model": os.environ.get("OLLAMA_MODEL", "llama3.2:3b"),
        "ollama_mode": os.environ.get("OLLAMA_MODE", "generate"),
    }

    # Optional config file to override defaults
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            file_cfg = json.load(f)
        config.update(file_cfg)

    for run_idx in range(args.runs):
        run_id = base_run_id if args.runs == 1 else f"{base_run_id}_{run_idx+1}"
        run_dir = os.path.join("data", "runs", run_id)
        os.makedirs(run_dir, exist_ok=True)

        if args.seed is not None:
            random.seed(args.seed + run_idx)

        flip_candidates = []
        for y in range(height):
            for x in range(width):
                cell = (x, y)
                if cell in (start, goal):
                    continue
                flip_candidates.append(cell)

        with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        # Precompute A* path for a reliable baseline
        astar_path_cells = astar_path(width, height, walls, start, goal)
        agents = [AStarAgent(astar_path_cells), RandomAgent(), GreedyAgent()]
        ollama_base_url = config["ollama_base_url"]
        ollama_model = config["ollama_model"]
        if ollama_base_url and ollama_model:
            provider = OllamaProvider(
                ollama_base_url,
                ollama_model,
                config["ollama_mode"],
                structured_output=config["structured_output"],
                timeout_s=config["llm_timeout"],
                retries=config["llm_retries"],
            )
            agents.append(LLMAgent(provider))

        summaries = []
        for agent in agents:
            env = GridEnv(width, height, walls, start, goal, flip_candidates=flip_candidates)
            env.reset()
            summary = run_agent(
                env,
                agent.name,
                agent.choose_action,
                run_dir,
                max_steps=config["max_steps"],
                history_steps=config["history_steps"],
                include_local_grid=config["local_grid"],
                local_grid_radius=config["local_grid_radius"],
                dynamic_walls=config["dynamic_walls"],
                flip_every=config["flip_every"],
                progress_every=config["progress_every"],
            )
            summaries.append(summary)

        print("Run complete")
        print(f"Output folder: {run_dir}")
        agent_list = ", ".join([s["agent"] for s in summaries]) or "none"
        print(f"Agents: {agent_list}")
        print("Summary:")
        for summary in summaries:
            print(
                f"- {summary['agent']}: steps={summary['steps']}, "
                f"invalid_move_rate={summary['invalid_move_rate']:.3f}, "
                f"loop_rate={summary['loop_rate']:.3f}, "
                f"fallback_rate={summary.get('fallback_rate', 0.0):.3f}, "
                f"reached_goal={summary.get('reached_goal')}, "
                f"final_distance={summary.get('final_distance')}, "
                f"min_distance={summary.get('min_distance')}"
            )
        print("Files:")
        print("- step_log_{agent}.jsonl")
        print("- summary_{agent}.json")
        print("- traj_{agent}.png")
        print("- heat_invalid_{agent}.png")
        print("- heat_loops_{agent}.png")


if __name__ == "__main__":
    main()
