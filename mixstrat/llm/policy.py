"""Read an LLM's strategy for a whole game as a ``Profile``.

Every infoset in the tree is shown to the backend once; the letter
distributions are mapped back to tree actions and assembled with
``Profile.from_infosets``. Answers can be cached on disk, since a prompt's
answer only depends on the backend and the prompt text.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mixstrat.game.strategy import Profile
from mixstrat.game.tree import FloatArray, GameTree, Infoset
from mixstrat.llm.backend import ChoiceDistribution, PolicyBackend, ReasoningSample
from mixstrat.llm.prompts import Message, Prompt, build_prompt

log = logging.getLogger(__name__)


def to_action_order(prompt: Prompt, letter_probs: FloatArray) -> FloatArray:
    """Reorder probabilities over the offered letters into tree action order."""
    probs = np.zeros(len(prompt.order))
    probs[list(prompt.order)] = letter_probs
    return probs


@dataclass(frozen=True)
class InfosetQuery:
    prompt: Prompt
    probs: FloatArray  # over the node's actions, in tree order
    legal_mass: float
    cached: bool
    samples: tuple[ReasoningSample, ...] = ()  # reasoning traces; probs in letter order

    @property
    def infoset(self) -> Infoset:
        return self.prompt.infoset

    def action_probs(self, letter_probs: FloatArray) -> FloatArray:
        return to_action_order(self.prompt, letter_probs)


@dataclass
class PolicyResult:
    profile: Profile
    queries: list[InfosetQuery]

    @property
    def min_legal_mass(self) -> float:
        return min(q.legal_mass for q in self.queries)

    @property
    def mean_legal_mass(self) -> float:
        return float(np.mean([q.legal_mass for q in self.queries]))

    @property
    def num_cached(self) -> int:
        return sum(q.cached for q in self.queries)

    @property
    def samples(self) -> list[ReasoningSample]:
        return [s for q in self.queries for s in q.samples]

    @property
    def truncated_rate(self) -> float:
        """Share of reasoning traces cut off at the token budget (0 without reasoning)."""
        samples = self.samples
        return float(np.mean([s.truncated for s in samples])) if samples else 0.0

    @property
    def mean_think_tokens(self) -> float:
        samples = self.samples
        return float(np.mean([s.tokens for s in samples])) if samples else 0.0


class ChoiceCache:
    """Append-only JSONL cache of backend answers."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._entries: dict[str, ChoiceDistribution] = {}
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    row = json.loads(line)
                    samples = tuple(
                        ReasoningSample(
                            s["thinking"],
                            np.asarray(s["probs"], dtype=np.float64),
                            s["legal_mass"],
                            s["tokens"],
                            s["truncated"],
                        )
                        for s in row.get("samples", [])
                    )
                    self._entries[row["key"]] = ChoiceDistribution(
                        np.asarray(row["probs"], dtype=np.float64), row["legal_mass"], samples
                    )

    @staticmethod
    def key(backend: str, messages: Sequence[Message], letters: Sequence[str]) -> str:
        blob = json.dumps([backend, list(messages), list(letters)], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str) -> ChoiceDistribution | None:
        return self._entries.get(key)

    def put(self, key: str, dist: ChoiceDistribution) -> None:
        self._entries[key] = dist
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row: dict[str, object] = {
            "key": key,
            "probs": dist.probs.tolist(),
            "legal_mass": dist.legal_mass,
        }
        if dist.samples:
            row["samples"] = [
                {
                    "thinking": s.thinking,
                    "probs": s.probs.tolist(),
                    "legal_mass": s.legal_mass,
                    "tokens": s.tokens,
                    "truncated": s.truncated,
                }
                for s in dist.samples
            ]
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")

    def __len__(self) -> int:
        return len(self._entries)


def query_profile(
    tree: GameTree,
    backend: PolicyBackend,
    *,
    shuffle: bool = False,
    seed: int = 0,
    reasoning: bool = False,
    cache: ChoiceCache | None = None,
    progress: Callable[[int, int, InfosetQuery], None] | None = None,
) -> PolicyResult:
    """Ask ``backend`` for its action distribution at every infoset of ``tree``.

    ``reasoning`` selects the prompt that asks for brief thinking; pair it with a
    backend that reasons. ``progress(done, total, query)`` runs after each infoset."""
    table: dict[Infoset, FloatArray] = {}
    queries: list[InfosetQuery] = []
    for i, infoset in enumerate(tree.infosets):
        prompt = build_prompt(tree, infoset, shuffle=shuffle, seed=seed, reasoning=reasoning)
        log.info(
            "[%d/%d] %s | %s", i + 1, len(tree.infosets), infoset, "   ".join(prompt.option_lines)
        )
        key = ChoiceCache.key(backend.name, prompt.messages, prompt.letters)
        dist = cache.get(key) if cache is not None else None
        cached = dist is not None
        if dist is None:
            dist = backend.choose(prompt.messages, prompt.letters)
            if cache is not None:
                cache.put(key, dist)
        probs = to_action_order(prompt, dist.probs)
        table[infoset] = probs
        query = InfosetQuery(prompt, probs, dist.legal_mass, cached, dist.samples)
        queries.append(query)
        labels = [a.label for a in tree.actions_at(infoset)]
        answer = " / ".join(f"{a} {p:.2f}" for a, p in zip(labels, probs, strict=True))
        log.info("    -> %s%s", answer, " (cached)" if cached else "")
        if progress is not None:
            progress(i + 1, len(tree.infosets), query)
    return PolicyResult(Profile.from_infosets(tree, table), queries)
