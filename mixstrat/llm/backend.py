"""Backends that score a lettered multiple-choice prompt.

A backend reads the model's next-token distribution after the prompt and
returns the probability of each offered letter: an exact action distribution,
with no sampling. Concrete backends live in their own modules (``ollama``) so
their client libraries stay optional.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from mixstrat.game.tree import FloatArray
from mixstrat.llm.prompts import Message


class BackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChoiceDistribution:
    probs: FloatArray  # over the offered letters, renormalized to sum to 1
    legal_mass: float  # probability the model put on the offered letters before renormalizing


class PolicyBackend(Protocol):
    @property
    def name(self) -> str:
        """Identifies the model and settings; part of the cache key."""
        ...

    def choose(self, messages: Sequence[Message], letters: Sequence[str]) -> ChoiceDistribution: ...


def letter_distribution(
    top_logprobs: Sequence[Mapping[str, Any]], letters: Sequence[str]
) -> ChoiceDistribution:
    """Distribution over ``letters`` from a list of ``{"token", "logprob"}`` candidates.

    Tokens are matched after stripping whitespace, so "A" and " A" both count
    for A. Letters missing from the candidates get probability 0.
    """
    mass = np.zeros(len(letters))
    for entry in top_logprobs:
        token = str(entry["token"]).strip()
        if token in letters:
            mass[letters.index(token)] += math.exp(entry["logprob"])
    total = float(mass.sum())
    if total <= 0:
        seen = [entry["token"] for entry in top_logprobs]
        raise BackendError(f"none of {list(letters)} among the model's top tokens {seen}")
    return ChoiceDistribution(mass / total, total)
