from __future__ import annotations

import os
import sys

# Allow running as a script from repo root.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox
import numpy as np

from agents.llm_agent import LLMAgent, OllamaProvider
from agents.base import Observation
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
    scroll_offset: int = 0


class InteractiveViewer:
    def __init__(self) -> None:
        plt.rcParams["keymap.save"] = []  # disable default 's' save binding
        maze = os.environ.get("MAZE", "simple")
        if maze == "hard":
            width, height, walls, start, goal = make_hard_maze()
        else:
            width, height, walls, start, goal = make_simple_maze()
        self.env = GridEnv(width, height, walls, start, goal)
        self.width = width
        self.height = height
        self.walls = set(walls)
        self.goal = goal

        base_url = os.environ.get("OLLAMA_BASE_URL", "http://100.121.2.67:11434")
        model = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
        mode = os.environ.get("OLLAMA_MODE", "generate")
        provider = OllamaProvider(base_url, model, mode, structured_output=False)
        self.agent = LLMAgent(provider)

        self.state = ViewerState(history_steps=10, local_grid=True, local_grid_radius=2, step=0, last_action=None, history=[])

        self.fig = plt.figure(figsize=(12, 6))
        gs = self.fig.add_gridspec(2, 3, width_ratios=[1, 1.4, 0.4], height_ratios=[1, 0.2])
        self.ax_grid = self.fig.add_subplot(gs[0, 0])
        self.ax_text = self.fig.add_subplot(gs[0, 1])
        self.ax_text.axis("off")
        self.ax_controls = self.fig.add_subplot(gs[0, 2])
        self.ax_controls.axis("off")
        self.ax_status = self.fig.add_subplot(gs[1, 0:2])
        self.ax_status.axis("off")

        self.running = False
        self.last_invalid = False
        self.prompt_offset = 0
        self.timer = None

        self.build_controls()
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("close_event", self.on_close)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.draw()

    def reset(self) -> None:
        self.env.reset()
        self.state.step = 0
        self.state.last_action = None
        self.state.history = []
        self.draw()

    def step_once(self) -> None:
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
            plan_prompt=False,
        )
        action_response = self.agent.choose_action(obs, prompt)
        action = action_response.parsed
        result = self.env.step(action)
        self.last_invalid = not result.valid_move

        self.state.history.append(f"Obs: {obs_text} | Action: {action}")
        self.state.last_action = action
        self.state.step += 1

        self.state.last_prompt = prompt
        self.state.last_raw = action_response.raw
        self.state.last_parsed = action_response.parsed
        self.draw()

    def toggle_run(self, event=None) -> None:
        self.running = not self.running
        if self.running:
            self.run_loop()

    def run_loop(self) -> None:
        if not self.running:
            return
        try:
            self.step_once()
        except Exception:
            self.running = False
            raise
        interval = max(0.1, float(self.speed_slider.val))
        if self.timer is None:
            self.timer = self.fig.canvas.new_timer(interval=interval * 1000)
            self.timer.add_callback(self.run_loop)
        self.timer.interval = interval * 1000
        self.timer.start()

    def draw(self) -> None:
        self.ax_grid.clear()
        grid = np.zeros((self.height, self.width))
        for x, y in self.walls:
            grid[y, x] = 1
        self.ax_grid.imshow(grid, cmap="Greys", origin="upper")
        self.ax_grid.scatter([self.env.pos[0]], [self.env.pos[1]], color="red", s=60, label="Agent")
        self.ax_grid.scatter([self.goal[0]], [self.goal[1]], color="gold", s=60, label="Goal")
        self.ax_grid.set_xticks(range(self.width))
        self.ax_grid.set_yticks(range(self.height))
        self.ax_grid.set_xlim(-0.5, self.width - 0.5)
        self.ax_grid.set_ylim(self.height - 0.5, -0.5)
        self.ax_grid.grid(True, linewidth=0.5, alpha=0.5)
        self.ax_grid.legend(loc="upper right", fontsize=8)

        self.ax_text.clear()
        self.ax_text.axis("off")
        prompt = self.state.last_prompt or ""
        raw = self.state.last_raw or ""
        parsed = self.state.last_parsed or ""
        lines = prompt.splitlines()
        max_lines = 40
        start = max(0, self.state.scroll_offset)
        end = min(len(lines), start + max_lines)
        prompt_view = "\n".join(lines[start:end])
        raw_view = raw if raw else "<empty>"
        text = (
            f"Step: {self.state.step}\n"
            f"History steps: {self.state.history_steps}\n"
            f"Local grid: {self.state.local_grid} (radius {self.state.local_grid_radius})\n"
            f"Last action: {self.state.last_action}\n\n"
            f"Parsed action: {parsed}\n"
            f"Raw output:\n{raw_view}\n\n"
            f"Prompt (line {start}):\n{prompt_view}\n"
        )
        self.ax_text.text(0.01, 0.99, text, va="top", ha="left", fontsize=7, family="monospace", clip_on=True)

        status = "INVALID MOVE" if self.last_invalid else "OK"
        color = "red" if self.last_invalid else "green"
        self.ax_status.clear()
        self.ax_status.axis("off")
        self.ax_status.text(0.01, 0.5, f"Status: {status}", color=color, fontsize=14, weight="bold")
        self.fig.canvas.draw_idle()

    def on_key(self, event) -> None:
        if event.key == "s":
            self.step_once()
        elif event.key == "r":
            self.reset()
        elif event.key == "g":
            self.state.local_grid = not self.state.local_grid
            self.draw()
        elif event.key == "h":
            self.state.history_steps = min(self.state.history_steps + 1, 50)
            self.draw()
        elif event.key == "H":
            self.state.history_steps = max(self.state.history_steps - 1, 0)
            self.draw()
        elif event.key == "k":
            self.state.local_grid_radius = min(self.state.local_grid_radius + 1, 5)
            self.draw()
        elif event.key == "K":
            self.state.local_grid_radius = max(self.state.local_grid_radius - 1, 1)
            self.draw()
        elif event.key == " ":
            self.toggle_run()

    def on_close(self, event) -> None:
        self.running = False
        if self.timer is not None:
            self.timer.stop()

    def on_scroll(self, event) -> None:
        # Scroll in text panel only
        if event.inaxes != self.ax_text:
            return
        delta = -1 if event.button == "up" else 1
        self.state.scroll_offset = max(0, self.state.scroll_offset + delta * 2)
        self.draw()

    def build_controls(self) -> None:
        # Buttons
        ax_step = self.fig.add_axes([0.77, 0.78, 0.18, 0.05])
        ax_run = self.fig.add_axes([0.77, 0.71, 0.18, 0.05])
        ax_reset = self.fig.add_axes([0.77, 0.64, 0.18, 0.05])
        ax_grid = self.fig.add_axes([0.77, 0.57, 0.18, 0.05])

        self.btn_step = Button(ax_step, "Step")
        self.btn_run = Button(ax_run, "Run/Pause")
        self.btn_reset = Button(ax_reset, "Reset")
        self.btn_grid = Button(ax_grid, "Toggle Grid")

        self.btn_step.on_clicked(lambda e: self.step_once())
        self.btn_run.on_clicked(self.toggle_run)
        self.btn_reset.on_clicked(lambda e: self.reset())
        self.btn_grid.on_clicked(lambda e: self._toggle_grid())

        # Sliders
        ax_speed = self.fig.add_axes([0.77, 0.48, 0.18, 0.03])
        self.speed_slider = Slider(ax_speed, "Speed (s)", 0.1, 2.0, valinit=0.5)

        ax_history = self.fig.add_axes([0.77, 0.41, 0.18, 0.03])
        self.history_slider = Slider(ax_history, "History", 0, 50, valinit=self.state.history_steps, valstep=1)
        self.history_slider.on_changed(self._set_history)

        ax_radius = self.fig.add_axes([0.77, 0.34, 0.18, 0.03])
        self.radius_slider = Slider(ax_radius, "Grid R", 1, 5, valinit=self.state.local_grid_radius, valstep=1)
        self.radius_slider.on_changed(self._set_radius)


    def _toggle_grid(self) -> None:
        self.state.local_grid = not self.state.local_grid
        self.draw()

    def _set_history(self, val) -> None:
        self.state.history_steps = int(val)
        self.draw()

    def _set_radius(self, val) -> None:
        self.state.local_grid_radius = int(val)
        self.draw()



def main() -> None:
    viewer = InteractiveViewer()
    print("Controls: s=step, r=reset, g=toggle grid, space=run/pause")
    plt.show()


if __name__ == "__main__":
    main()
