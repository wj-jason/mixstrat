"""``PolicyBackend`` for a model served by a local Ollama server.

"Direct" mode: thinking off, generate one token, and read the logprobs of the
answer letters from ``top_logprobs``. Ollama reports the model's raw logprobs
(they don't depend on temperature or top-p), so the result measures the model,
not the sampler.

Left alone, chat models open with markdown or prose ("**", "You have ..."), so
the first token says nothing about the choice. The assistant turn is therefore
prefilled up to the letter (default ``"Answer: **"``, matching how qwen3 formats
its answers). When Ollama continues a prefilled turn it drops the empty thinking
block it would add for ``think: false``, leaving the model mid-thought; for
models with the "thinking" capability the prefill closes that block itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import TracebackType

import httpx

from mixstrat.llm.backend import BackendError, ChoiceDistribution, letter_distribution
from mixstrat.llm.prompts import Message

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_PREFILL = "Answer: **"
EMPTY_THINKING = "<think>\n\n</think>\n\n"


class OllamaBackend:
    def __init__(
        self,
        model: str,
        host: str = DEFAULT_HOST,
        top_logprobs: int = 20,
        timeout: float = 300.0,
        keep_alive: str = "5m",
        prefill: str = DEFAULT_PREFILL,
    ) -> None:
        self.model = model
        self.top_logprobs = top_logprobs
        self.keep_alive = keep_alive
        self.prefill = prefill
        self._client = httpx.Client(base_url=host, timeout=timeout)
        self.thinking = "thinking" in self._capabilities()

    def _capabilities(self) -> list[str]:
        try:
            response = self._client.post("/api/show", json={"model": self.model})
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise BackendError(f"Ollama model {self.model!r} unavailable: {e}") from e
        return list(response.json().get("capabilities", []))

    @property
    def name(self) -> str:
        return f"ollama/{self.model}/direct/prefill={self.prefill!r}"

    def choose(self, messages: Sequence[Message], letters: Sequence[str]) -> ChoiceDistribution:
        payload = {
            "model": self.model,
            "messages": [*messages, {"role": "assistant", "content": self._prefix()}],
            "stream": False,
            "think": False,
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
            "keep_alive": self.keep_alive,
            "options": {"num_predict": 1, "temperature": 0},
        }
        try:
            response = self._client.post("/api/chat", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise BackendError(f"Ollama request failed: {e}") from e
        logprobs = response.json().get("logprobs")
        if not logprobs:
            raise BackendError("Ollama returned no logprobs; this server may not support them")
        return letter_distribution(logprobs[0]["top_logprobs"], letters)

    def _prefix(self) -> str:
        return (EMPTY_THINKING if self.thinking else "") + self.prefill

    def unload(self) -> None:
        """Free the model's GPU memory now instead of after ``keep_alive``."""
        self._client.post("/api/generate", json={"model": self.model, "keep_alive": 0})

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OllamaBackend:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
