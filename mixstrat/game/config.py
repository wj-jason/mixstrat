from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

Payoff = Literal["high", "low"]
_PAYOFFS: tuple[str, ...] = ("high", "low")


@dataclass(frozen=True)
class GameConfig:
    """Complete rules of one one-street poker variant."""

    name: str
    deck: tuple[int, ...] = (1, 1, 1)
    deck_p2: tuple[int, ...] | None = None
    antes: tuple[float, float] = (1.0, 1.0)
    bet_sizes: tuple[float, ...] = (0.5,)
    raise_sizes: tuple[float, ...] | None = None
    max_raises: int = 0
    first_actor: int = 0
    payoff: Payoff = "high"
    informed: tuple[bool, bool] = (True, True)
    description: str = ""

    def __post_init__(self) -> None:
        # Accept lists (e.g. from TOML) and store tuples so configs stay hashable.
        for f in ("deck", "deck_p2", "antes", "bet_sizes", "raise_sizes", "informed"):
            value = getattr(self, f)
            if value is not None and not isinstance(value, tuple):
                object.__setattr__(self, f, tuple(value))
        self._validate()

    # ------------------------------------------------------------ validation

    def _validate(self) -> None:
        def fail(msg: str) -> None:
            raise ValueError(f"GameConfig '{self.name}': {msg}")

        if not self.name:
            fail("name must be non-empty")

        for label, deck in (("deck", self.deck), ("deck_p2", self.deck_p2)):
            if deck is None:
                continue
            if len(deck) < 2:
                fail(f"{label} needs at least 2 ranks")
            if any(not isinstance(c, int) or c < 0 for c in deck):
                fail(f"{label} counts must be non-negative integers")
            if sum(deck) == 0:
                fail(f"{label} has no cards")

        if self.deck_p2 is None:
            if sum(self.deck) < 2:
                fail("a shared deck needs at least 2 cards")
        elif len(self.deck_p2) != len(self.deck):
            fail("deck and deck_p2 must have the same number of ranks")

        if len(self.antes) != 2 or any(a <= 0 for a in self.antes):
            fail("antes must be two positive numbers")
        if not self.bet_sizes or any(s <= 0 for s in self.bet_sizes):
            fail("bet_sizes must be a non-empty list of positive pot fractions")
        if len(set(self.bet_sizes)) != len(self.bet_sizes):
            fail("bet_sizes must not repeat")
        if self.raise_sizes is not None and (
            not self.raise_sizes or any(s <= 0 for s in self.raise_sizes)
        ):
            fail("raise_sizes must be a non-empty list of positive pot fractions")
        if self.max_raises < 0:
            fail("max_raises must be >= 0")
        if self.first_actor not in (0, 1):
            fail("first_actor must be 0 or 1")
        if self.payoff not in _PAYOFFS:
            fail(f"payoff must be one of {_PAYOFFS}")
        if len(self.informed) != 2:
            fail("informed must have one entry per player")

    # ------------------------------------------------------------ derived

    @property
    def num_ranks(self) -> int:
        return len(self.deck)

    @property
    def shared_deck(self) -> bool:
        return self.deck_p2 is None

    @property
    def initial_pot(self) -> float:
        return float(sum(self.antes))

    @property
    def effective_raise_sizes(self) -> tuple[float, ...]:
        return self.raise_sizes if self.raise_sizes is not None else self.bet_sizes

    def deal_distribution(self) -> npt.NDArray[np.float64]:
        """Joint deal probabilities: ``P[i, j]`` = P(player 0 has rank i, player 1 has rank j)."""
        c1 = np.asarray(self.deck, dtype=np.float64)
        if self.deck_p2 is not None:
            c2 = np.asarray(self.deck_p2, dtype=np.float64)
            return np.outer(c1 / c1.sum(), c2 / c2.sum())
        total = c1.sum()
        # Second card drawn without replacement: one fewer copy of the first card's rank.
        joint = np.outer(c1, c1) - np.diag(c1)
        return joint / (total * (total - 1))

    # ------------------------------------------------------------ construction

    @classmethod
    def uniform(cls, name: str, num_ranks: int, copies: int = 1, **kwargs: Any) -> GameConfig:
        """Shared deck with ``copies`` of each of ``num_ranks`` ranks."""
        return cls(name=name, deck=(copies,) * num_ranks, **kwargs)

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> GameConfig:
        """Build from a plain dict, e.g. one TOML table.

        Besides the dataclass fields, accepts the shorthand ``num_ranks`` (+ optional
        ``copies``) in place of ``deck``.
        """
        data = dict(data)
        allowed = {f.name for f in fields(cls)} - {"name"}
        if "num_ranks" in data:
            if "deck" in data:
                raise ValueError(f"GameConfig '{name}': give either deck or num_ranks, not both")
            data["deck"] = (int(data.pop("copies", 1)),) * int(data.pop("num_ranks"))
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"GameConfig '{name}': unknown keys {sorted(unknown)}")
        return cls(name=name, **data)

    def to_dict(self) -> dict[str, Any]:
        """Plain dict of all fields, for logging alongside results."""
        return asdict(self)


def load_games(path: str | Path) -> dict[str, GameConfig]:
    """Load every game defined in a TOML file; each top-level table is one game."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return {name: GameConfig.from_dict(name, table) for name, table in raw.items()}
