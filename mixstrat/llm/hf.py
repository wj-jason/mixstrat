"""``PolicyBackend`` for a Hugging Face model: the training backend.

Direct mode only. The prompt is rendered with the model's chat template
(``enable_thinking=False`` adds the same empty thinking block Ollama uses),
followed by the same ``Answer: **`` prefill, and the answer letters' log
probabilities are read from the full next-token distribution (no top-k
cut-off).

By default the text also matches Ollama's own qwen3 template, which differs
from the Hugging Face one in two ways: it appends " /no_think" (Qwen3's in-prompt
switch) to the last user message, and it leaves a blank line after the system
header. At close decisions that difference alone moves answer probabilities by
up to ~0.5, so matching it keeps HF results comparable with the Ollama
baselines. ``letter_logprobs`` is the differentiable version used for training;
``choose`` wraps it for evaluation, so a model being trained can be scored with
``query_profile`` like any other backend.

Unlike Ollama's Q4 GGUF build, the weights here are bf16, so the two backends
agree only up to quantization noise.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import numpy as np
import torch

from mixstrat.llm.backend import BackendError, ChoiceDistribution
from mixstrat.llm.prompts import DEFAULT_PREFILL, Message

DEFAULT_MODEL = "Qwen/Qwen3-1.7B"


class Float32Head(torch.autograd.Function):
    """``hidden @ weight.T`` with a float32 result from bf16 inputs, no fp32 weight copy.

    PyTorch has no backward for ``mm(..., out_dtype=...)``, so the backward is
    written out: only the gradient for ``hidden`` is returned, because the output
    weights stay frozen (LoRA trains adapters, and Qwen3 ties them to the input
    embeddings).
    """

    @staticmethod
    def forward(ctx: Any, hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(weight)
        return torch.mm(hidden, weight.T, out_dtype=torch.float32)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        (weight,) = ctx.saved_tensors
        return torch.mm(grad.to(weight.dtype), weight), None


@dataclass
class LetterLogprobs:
    logprobs: torch.Tensor  # over the offered letters, normalized among them
    legal_mass: torch.Tensor  # probability on the offered letters before normalizing
    log_legal_mass: torch.Tensor  # its log, computed stably


def letter_logprobs_from_row(
    row: torch.Tensor, variant_ids: Sequence[Sequence[int]]
) -> LetterLogprobs:
    """Letter log probabilities from one next-token log-softmax row.

    ``variant_ids[i]`` lists the token ids that spell letter ``i`` (e.g. "A" and
    " A"); their probabilities are added, matching ``letter_distribution``.
    """
    per_letter = torch.stack([torch.logsumexp(row[list(ids)], dim=0) for ids in variant_ids])
    total = torch.logsumexp(per_letter, dim=0)
    return LetterLogprobs(per_letter - total, total.exp(), total)


class HFBackend:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        model_id: str,
        prefill: str = DEFAULT_PREFILL,
        ollama_format: bool = True,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.prefill = prefill
        self.ollama_format = ollama_format
        self.tokenizer.padding_side = "left"
        self._variants: dict[str, list[int]] = {}

    @classmethod
    def load(
        cls,
        model_id: str = DEFAULT_MODEL,
        adapter: str | None = None,
        prefill: str = DEFAULT_PREFILL,
        device: str = "cuda",
        ollama_format: bool = True,
    ) -> HFBackend:
        """Load bf16 weights, optionally with a trained LoRA adapter on top."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Any: a plain model or a LoRA-wrapped one; both are called the same way.
        model: Any = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device
        )
        name = model_id
        if adapter is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
            name = f"{model_id}+{adapter}"
        model.eval()
        return cls(model, tokenizer, name, prefill, ollama_format)

    @property
    def name(self) -> str:
        fmt = "/ollama-format" if self.ollama_format else ""
        return f"hf/{self.model_id}/direct/prefill={self.prefill!r}{fmt}"

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def render(self, messages: Sequence[Message]) -> str:
        messages = list(messages)
        if self.ollama_format:
            last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
            content = messages[last_user]["content"] + " /no_think"
            messages[last_user] = {**messages[last_user], "content": content}
        text: str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        if self.ollama_format:
            text = text.replace("<|im_start|>system\n", "<|im_start|>system\n\n", 1)
        return text + self.prefill

    def _variant_ids(self, letter: str) -> list[int]:
        if letter not in self._variants:
            ids = []
            for spelling in (letter, f" {letter}"):
                tokens = self.tokenizer.encode(spelling, add_special_tokens=False)
                if len(tokens) == 1:
                    ids.append(tokens[0])
            if not ids:
                raise BackendError(f"letter {letter!r} is not a single token for {self.model_id}")
            self._variants[letter] = ids
        return self._variants[letter]

    def letter_logprobs(
        self, batch: Sequence[tuple[Sequence[Message], Sequence[str]]]
    ) -> list[LetterLogprobs]:
        """Differentiable letter log probabilities for a batch of (messages, letters)."""
        texts = [self.render(messages) for messages, _ in batch]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        # Only the last position is needed (left padding keeps it real); computing the
        # full-vocabulary logits for every position would take ~1.5 GB for kuhn3.
        out = self.model(**inputs, output_hidden_states=True, logits_to_keep=1)
        # Recompute that position's logits with a float32 result. The model's own bf16
        # logits are rounded to multiples of 0.125-0.25 at typical magnitudes, which
        # quantizes answer probabilities (a 1/3 mix can only be 0.32 or 0.38) and swallows
        # small training updates.
        hidden = out.hidden_states[-1][:, -1, :]
        weight = self.model.get_output_embeddings().weight
        logits = Float32Head.apply(hidden, weight)
        rows = torch.log_softmax(logits, dim=-1)
        return [
            letter_logprobs_from_row(row, [self._variant_ids(letter) for letter in letters])
            for row, (_, letters) in zip(rows, batch, strict=True)
        ]

    def choose(self, messages: Sequence[Message], letters: Sequence[str]) -> ChoiceDistribution:
        with torch.no_grad():
            (out,) = self.letter_logprobs([(messages, letters)])
        return ChoiceDistribution(
            out.logprobs.exp().cpu().numpy().astype(np.float64), float(out.legal_mass)
        )

    def close(self) -> None:
        self.model = None
        torch.cuda.empty_cache()

    def __enter__(self) -> HFBackend:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
