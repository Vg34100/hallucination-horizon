from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass

from agents.ollama_client import OllamaClient


@dataclass
class LLMResponse:
    raw: str
    parsed: str
    fallback_used: bool


class OllamaProvider:
    # Handles prompt construction + Ollama calls + response parsing.
    def __init__(
        self,
        base_url: str,
        model: str,
        mode: str = "generate",
        structured_output: bool = False,
        timeout_s: int = 60,
        retries: int = 1,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.mode = mode
        self.structured_output = structured_output
        self.timeout_s = timeout_s
        self.retries = retries
        self.num_ctx = 65536
        self.client = OllamaClient(self.base_url)

    def generate(
        self, prompt: str, max_tokens: int = 80, temperature: float = 0.5
    ) -> str:
        # Main LLM call (chat or generate). Returns raw text.
        if self.mode == "chat":
            options = {
                "num_predict": max_tokens,
                "temperature": temperature,
                "num_ctx": self.num_ctx,
            }
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "options": options,
                "stream": False,
            }
            data = None
            for _ in range(self.retries + 1):
                try:
                    data = self.client.chat(payload, self.timeout_s)
                    break
                except TimeoutError:
                    continue
            if data is None:
                return ""
            message = data.get("message", {})
            return message.get("content", "") or message.get("thinking", "") or ""

        fmt = None
        if self.structured_output:
            fmt = {
                "type": "object",
                "properties": {"action": {"type": "string", "enum": ["N", "S", "E", "W"]}},
                "required": ["action"],
            }

        prompt_block = f"{prompt}"
        options = {
            "num_predict": max_tokens,
            "temperature": temperature,
        }
        options["num_ctx"] = self.num_ctx
        payload = {
            "model": self.model,
            "prompt": prompt_block,
            "raw": False if self.structured_output else True,
            "options": options,
            "format": fmt,
            "stream": False,
        }
        data = None
        for _ in range(self.retries + 1):
            try:
                data = self.client.generate(payload, self.timeout_s)
                break
            except TimeoutError:
                continue
        if data is None:
            return ""
        return data.get("response", "")


class LLMAgent:
    name = "llm"

    def __init__(self, provider: OllamaProvider) -> None:
        self.provider = provider

    def choose_action(self, obs, prompt: str) -> LLMResponse:
        # Ask the model, then parse the final direction token.
        raw = self.provider.generate(prompt)
        parsed = ""
        if self.provider.structured_output:
            try:
                data = json.loads(raw)
                parsed = data.get("action", "")
            except json.JSONDecodeError:
                parsed = ""
        if not parsed:
            # Prefer the last standalone direction token (often comes at end).
            tokens = re.findall(r"\b([NSEW])\b", raw.upper())
            if tokens:
                parsed = tokens[-1]

        fallback_used = False
        if not parsed:
            # If model output is unusable, fall back to a valid random move.
            open_actions = [a for a, is_open in obs.open_map.items() if is_open]
            if open_actions:
                parsed = random.choice(open_actions)
                fallback_used = True

        return LLMResponse(raw=raw, parsed=parsed, fallback_used=fallback_used)
