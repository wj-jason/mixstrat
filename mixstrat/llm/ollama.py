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

"Reasoning" mode: sample ``samples`` reasoning traces with thinking on, each
capped at ``think_tokens`` and drawn with the model's own sampling settings
(seeded, so runs repeat). Each trace is then fed back inside the thinking
block, followed by the same prefill, and the answer letters are read exactly
as in direct mode. A trace cut off at the budget is closed where it stopped.
The reported distribution is the average over traces, so it reflects both the
spread of reasoning and the model's uncertainty after each trace.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from types import TracebackType
from typing import Any, Literal

import httpx
import numpy as np

from mixstrat.llm.backend import (
    BackendError,
    ChoiceDistribution,
    ReasoningSample,
    letter_distribution,
)
from mixstrat.llm.prompts import DEFAULT_PREFILL, Message

DEFAULT_HOST = "http://localhost:11434"
EMPTY_THINKING = "<think>\n\n</think>\n\n"

Mode = Literal["direct", "reasoning"]

log = logging.getLogger(__name__)


class OllamaBackend:
    def __init__(
        self,
        model: str,
        host: str = DEFAULT_HOST,
        top_logprobs: int = 20,
        timeout: float = 300.0,
        keep_alive: str = "5m",
        prefill: str = DEFAULT_PREFILL,
        mode: Mode = "direct",
        samples: int = 4,
        think_tokens: int = 1024,
        seed: int = 0,
        num_ctx: int | None = None,
    ) -> None:
        self.model = model
        self.num_ctx = num_ctx  # context window; None keeps Ollama's default (4096)
        self.top_logprobs = top_logprobs
        self.keep_alive = keep_alive
        self.prefill = prefill
        self.mode = mode
        self.samples = samples
        self.think_tokens = think_tokens
        self.seed = seed
        self._client = httpx.Client(base_url=host, timeout=timeout)
        self.thinking = "thinking" in self._capabilities()
        if mode == "reasoning" and not self.thinking:
            raise BackendError(f"reasoning mode needs a thinking model; {model!r} isn't one")

    def _capabilities(self) -> list[str]:
        try:
            response = self._client.post("/api/show", json={"model": self.model})
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise BackendError(f"Ollama model {self.model!r} unavailable: {e}") from e
        return list(response.json().get("capabilities", []))

    @property
    def name(self) -> str:
        ctx = f"/num_ctx={self.num_ctx}" if self.num_ctx is not None else ""
        if self.mode == "direct":
            return f"ollama/{self.model}/direct/prefill={self.prefill!r}{ctx}"
        return (
            f"ollama/{self.model}/reasoning/samples={self.samples}/think_tokens={self.think_tokens}"
            f"/seed={self.seed}/prefill={self.prefill!r}{ctx}"
        )

    def _chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = {"model": self.model, "stream": False, "keep_alive": self.keep_alive, **payload}
        if self.num_ctx is not None:
            # Every call must send the same num_ctx, or Ollama reloads the model.
            payload["options"] = {**payload.get("options", {}), "num_ctx": self.num_ctx}
        try:
            response = self._client.post("/api/chat", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise BackendError(f"Ollama request failed: {e}") from e
        data: dict[str, Any] = response.json()
        return data

    def _letters(
        self, messages: Sequence[Message], prefix: str, letters: Sequence[str]
    ) -> ChoiceDistribution:
        """Letter distribution right after an assistant turn that starts with ``prefix``."""
        data = self._chat(
            {
                "messages": [*messages, {"role": "assistant", "content": prefix}],
                "think": False,
                "logprobs": True,
                "top_logprobs": self.top_logprobs,
                "options": {"num_predict": 1, "temperature": 0},
            }
        )
        logprobs = data.get("logprobs")
        if not logprobs:
            raise BackendError("Ollama returned no logprobs; this server may not support them")
        return letter_distribution(logprobs[0]["top_logprobs"], letters)

    def _reason(self, messages: Sequence[Message], index: int) -> tuple[str, int, bool]:
        data = self._chat(
            {
                "messages": list(messages),
                "think": True,
                "options": {"num_predict": self.think_tokens, "seed": self.seed * 1000 + index},
            }
        )
        message = data["message"]
        finished = bool(message.get("content", "").strip())
        truncated = data.get("done_reason") == "length" and not finished
        return message.get("thinking", "").strip(), int(data.get("eval_count", 0)), truncated

    def choose(self, messages: Sequence[Message], letters: Sequence[str]) -> ChoiceDistribution:
        if self.mode == "direct":
            prefix = (EMPTY_THINKING if self.thinking else "") + self.prefill
            return self._letters(messages, prefix, letters)

        samples = []
        for i in range(self.samples):
            thinking, tokens, truncated = self._reason(messages, i)
            prefix = f"<think>\n{thinking}\n</think>\n\n{self.prefill}"
            dist = self._letters(messages, prefix, letters)
            samples.append(
                ReasoningSample(thinking, dist.probs, dist.legal_mass, tokens, truncated)
            )
            answer = " / ".join(f"{L} {p:.2f}" for L, p in zip(letters, dist.probs, strict=True))
            cut = " (cut off)" if truncated else ""
            log.info(
                "    sample %d/%d: %d tokens%s -> %s", i + 1, self.samples, tokens, cut, answer
            )
            log.debug("      %s", thinking.replace("\n", "\n      "))
        return ChoiceDistribution(
            np.mean([s.probs for s in samples], axis=0),
            float(np.mean([s.legal_mass for s in samples])),
            tuple(samples),
        )

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
