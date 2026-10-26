"""Read an LLM's strategy for a whole game as a ``Profile``.

Every infoset in the tree is shown to the backend once; the letter
distributions are mapped back to tree actions and assembled with
``Profile.from_infosets``. Answers can be cached on disk, since a prompt's
answer only depends on the backend and the prompt text.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mixstrat.game.strategy import Profile
from mixstrat.game.tree import FloatArray, GameTree, Infoset
from mixstrat.llm.backend import ChoiceDistribution, PolicyBackend
from mixstrat.llm.prompts import Message, Prompt, build_prompt


@dataclass(frozen=True)
class InfosetQuery:
    prompt: Prompt
    probs: FloatArray  # over the node's actions, in tree order
    legal_mass: float
    cached: bool

    @property
    def infoset(self) -> Infoset:
        return self.prompt.infoset


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


class ChoiceCache:
    """Append-only JSONL cache of backend answers."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._entries: dict[str, ChoiceDistribution] = {}
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    row = json.loads(line)
                    self._entries[row["key"]] = ChoiceDistribution(
                        np.asarray(row["probs"], dtype=np.float64), row["legal_mass"]
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
        row = {"key": key, "probs": dist.probs.tolist(), "legal_mass": dist.legal_mass}
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
    cache: ChoiceCache | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> PolicyResult:
    """Ask ``backend`` for its action distribution at every infoset of ``tree``."""
    table: dict[Infoset, FloatArray] = {}
    queries: list[InfosetQuery] = []
    for i, infoset in enumerate(tree.infosets):
        prompt = build_prompt(tree, infoset, shuffle=shuffle, seed=seed)
        key = ChoiceCache.key(backend.name, prompt.messages, prompt.letters)
        dist = cache.get(key) if cache is not None else None
        cached = dist is not None
        if dist is None:
            dist = backend.choose(prompt.messages, prompt.letters)
            if cache is not None:
                cache.put(key, dist)
        probs = np.zeros(len(prompt.order))
        probs[list(prompt.order)] = dist.probs
        table[infoset] = probs
        queries.append(InfosetQuery(prompt, probs, dist.legal_mass, cached))
        if progress is not None:
            progress(i + 1, len(tree.infosets))
    return PolicyResult(Profile.from_infosets(tree, table), queries)
