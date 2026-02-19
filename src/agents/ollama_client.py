from __future__ import annotations

import json
import urllib.request


class OllamaClient:
    # Thin HTTP wrapper around Ollama.
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def chat(self, payload: dict, timeout_s: int) -> dict:
        req = urllib.request.Request(
            url=f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def generate(self, payload: dict, timeout_s: int) -> dict:
        req = urllib.request.Request(
            url=f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
