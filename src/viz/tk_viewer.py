from __future__ import annotations

import os
import sys

# Allow running as a script from repo root.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
import queue
import threading
import urllib.request
from dataclasses import dataclass
from typing import List, Tuple

import tkinter as tk
from tkinter import ttk

from agents.observation import Observation
from agents.llm_agent import LLMAgent, OllamaProvider
from core.env_grid import GridEnv
from core.mazes import make_hard_maze, make_simple_maze
from prompts.prompts import build_prompt, format_local_grid, format_observation


Coord = Tuple[int, int]


@dataclass
class ViewerState:
    history_steps: int = 10
    local_grid: bool = True
    local_grid_radius: int = 2
    step: int = 0
    last_action: str | None = None
    history: List[str] = None
    last_prompt: str = ""
    last_raw: str = ""
    last_parsed: str = ""


class TkViewer:
    def __init__(self) -> None:
        # Main UI entry point.
        self.root = tk.Tk()
        self.root.title("LLM Grid Viewer")

        self.base_url = tk.StringVar(value=os.environ.get("OLLAMA_BASE_URL", "http://100.121.2.67:11434"))
        self.model_name = tk.StringVar(value=os.environ.get("OLLAMA_MODEL", "llama3.2:3b"))
        self.maze_name = tk.StringVar(value="simple")
        self.running = False
        self.step_in_progress = False
        self.task_queue: queue.Queue = queue.Queue()

        self.state = ViewerState(history_steps=10, local_grid=True, local_grid_radius=2, step=0, last_action=None, history=[])
        self.init_env()
        self.init_agent()

        self.build_ui()
        self.draw()

        self.root.after(50, self.poll_queue)

    def init_env(self) -> None:
        # Build environment based on current maze choice.
        if self.maze_name.get() == "hard":
            width, height, walls, start, goal = make_hard_maze()
        else:
            width, height, walls, start, goal = make_simple_maze()
        self.env = GridEnv(width, height, walls, start, goal)
        self.width = width
        self.height = height
        self.walls = set(walls)
        self.goal = goal
        self.env.reset()

    def init_agent(self) -> None:
        # Rebuild LLM client when model/base URL changes.
        provider = OllamaProvider(
            self.base_url.get(),
            self.model_name.get(),
            "chat",
            structured_output=True,
        )
        self.agent = LLMAgent(provider)

    def build_ui(self) -> None:
        # Layout: top controls, grid canvas, and scrollable text panel.
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=8, pady=4)

        ttk.Label(top, text="Base URL").pack(side="left")
        ttk.Entry(top, textvariable=self.base_url, width=30).pack(side="left", padx=4)

        ttk.Label(top, text="Model").pack(side="left")
        self.model_box = ttk.Combobox(top, textvariable=self.model_name, width=20)
        self.model_box.pack(side="left", padx=4)
        ttk.Button(top, text="Refresh", command=self.refresh_models).pack(side="left", padx=4)

        ttk.Label(top, text="Maze").pack(side="left")
        self.maze_box = ttk.Combobox(top, textvariable=self.maze_name, values=["simple", "hard"], width=8)
        self.maze_box.pack(side="left", padx=4)
        ttk.Button(top, text="Set Maze", command=self.set_maze).pack(side="left", padx=4)

        controls = ttk.Frame(self.root)
        controls.pack(fill="x", padx=8, pady=4)
        ttk.Button(controls, text="Step", command=self.step_once).pack(side="left", padx=4)
        ttk.Button(controls, text="Run/Pause", command=self.toggle_run).pack(side="left", padx=4)
        ttk.Button(controls, text="Reset", command=self.reset).pack(side="left", padx=4)

        ttk.Label(controls, text="History").pack(side="left", padx=4)
        self.history_spin = ttk.Spinbox(controls, from_=0, to=50, width=4, command=self.update_history)
        self.history_spin.set(self.state.history_steps)
        self.history_spin.pack(side="left")

        ttk.Label(controls, text="Grid R").pack(side="left", padx=4)
        self.grid_spin = ttk.Spinbox(controls, from_=1, to=5, width=4, command=self.update_grid_radius)
        self.grid_spin.set(self.state.local_grid_radius)
        self.grid_spin.pack(side="left")

        ttk.Checkbutton(controls, text="Local grid", command=self.toggle_grid, variable=tk.BooleanVar(value=True)).pack(side="left", padx=4)

        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=8, pady=4)

        self.canvas = tk.Canvas(main, width=360, height=360, bg="white")
        self.canvas.pack(side="left", fill="both", expand=False, padx=4)

        text_frame = ttk.Frame(main)
        text_frame.pack(side="left", fill="both", expand=True)

        self.text = tk.Text(text_frame, wrap="none", width=60)
        self.text.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(text_frame, command=self.text.yview)
        scroll.pack(side="right", fill="y")
        self.text.config(yscrollcommand=scroll.set)

        self.status = ttk.Label(self.root, text="Status: OK")
        self.status.pack(fill="x", padx=8, pady=4)

    def refresh_models(self) -> None:
        # Query Ollama for available models and refresh dropdown.
        try:
            req = urllib.request.Request(
                url=f"{self.base_url.get()}/api/tags",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            models = [m["name"] for m in data.get("models", [])]
            if models:
                self.model_box["values"] = models
                self.model_name.set(models[0])
                self.init_agent()
        except Exception:
            pass

    def set_maze(self) -> None:
        self.init_env()
        self.reset()

    def update_history(self) -> None:
        self.state.history_steps = int(self.history_spin.get())

    def update_grid_radius(self) -> None:
        self.state.local_grid_radius = int(self.grid_spin.get())

    def toggle_grid(self) -> None:
        self.state.local_grid = not self.state.local_grid

    def reset(self) -> None:
        # Reset agent state + UI.
        self.running = False
        self.env.reset()
        self.state.step = 0
        self.state.last_action = None
        self.state.history = []
        self.state.last_prompt = ""
        self.state.last_raw = ""
        self.state.last_parsed = ""
        self.draw()

    def step_once(self) -> None:
        # Run one step in a background thread.
        if self.step_in_progress:
            return
        self.step_in_progress = True
        threading.Thread(target=self._step_worker, daemon=True).start()

    def _step_worker(self) -> None:
        # Single LLM step: build prompt -> query -> act -> update state.
        open_map = self.env.neighbors_open(self.env.pos)
        obs = Observation(
            pos=self.env.pos,
            goal=self.env.goal,
            open_map=open_map,
            last_action=self.state.last_action,
            step=self.state.step,
        )
        obs_text = format_observation(
            obs.pos, obs.goal, obs.open_map, obs.last_action, obs.step
        )
        local_grid = None
        if self.state.local_grid:
            local_grid = format_local_grid(
                obs.pos,
                obs.goal,
                self.env.walls,
                self.env.width,
                self.env.height,
                self.state.local_grid_radius,
            )
        prompt = build_prompt(
            obs_text,
            self.state.history[-self.state.history_steps :],
            local_grid,
        )
        action_response = self.agent.choose_action(obs, prompt)
        print(f"Raw response: '{action_response.raw}'")
        print(f"Parsed: '{action_response.parsed}'")
        print(f"Fallback used: {action_response.fallback_used}")
        action = action_response.parsed
        result = self.env.step(action)

        self.state.history.append(f"Obs: {obs_text} | Action: {action}")
        self.state.last_action = action
        self.state.step += 1
        self.state.last_prompt = prompt
        self.state.last_raw = action_response.raw
        self.state.last_parsed = action_response.parsed

        if result.pos_after == self.env.goal:
            self.running = False

        self.task_queue.put((result.valid_move, result.pos_after))

    def poll_queue(self) -> None:
        # UI loop: pull completed steps and update UI.
        try:
            valid_move, _ = self.task_queue.get_nowait()
            self.step_in_progress = False
            self.status.config(text=f"Status: {'OK' if valid_move else 'INVALID MOVE'}")
            self.draw()
        except queue.Empty:
            pass
        if self.running and not self.step_in_progress:
            self.step_once()
        self.root.after(50, self.poll_queue)

    def toggle_run(self) -> None:
        # Start/stop continuous stepping.
        self.running = not self.running

    def draw(self) -> None:
        # Redraw grid and text panel.
        self.canvas.delete("all")
        cell = 40
        for y in range(self.height):
            for x in range(self.width):
                x0, y0 = x * cell, y * cell
                x1, y1 = x0 + cell, y0 + cell
                fill = "white"
                if (x, y) in self.walls:
                    fill = "black"
                self.canvas.create_rectangle(x0, y0, x1, y1, outline="gray", fill=fill)
        ax, ay = self.env.pos
        gx, gy = self.goal
        self.canvas.create_oval(ax * cell + 10, ay * cell + 10, ax * cell + 30, ay * cell + 30, fill="red")
        self.canvas.create_oval(gx * cell + 10, gy * cell + 10, gx * cell + 30, gy * cell + 30, fill="gold")

        self.text.delete("1.0", tk.END)
        self.text.insert(
            tk.END,
            f"Step: {self.state.step}\n"
            f"History steps: {self.state.history_steps}\n"
            f"Local grid: {self.state.local_grid} (radius {self.state.local_grid_radius})\n"
            f"Last action: {self.state.last_action}\n\n"
            f"Parsed action: {self.state.last_parsed}\n\n"
            f"Raw output:\n{self.state.last_raw}\n\n"
            f"Prompt:\n{self.state.last_prompt}\n"
        )


def main() -> None:
    viewer = TkViewer()
    viewer.refresh_models()
    viewer.root.mainloop()


if __name__ == "__main__":
    main()
