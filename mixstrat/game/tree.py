from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from mixstrat.game.config import GameConfig

FloatArray = npt.NDArray[np.float64]
ActionKind = Literal["check", "bet", "fold", "call", "raise"]
TerminalKind = Literal["fold", "showdown"]


@dataclass(frozen=True)
class Action:
    """One betting action. ``size`` is a pot fraction for bets and raises, else None."""

    kind: ActionKind
    size: float | None = None

    @property
    def label(self) -> str:
        return self.kind if self.size is None else f"{self.kind}:{self.size:g}"

    @property
    def aggressive(self) -> bool:
        return self.kind in ("bet", "raise")

    def __str__(self) -> str:
        return self.label


@dataclass
class Node:
    """A node of the betting tree. Decision nodes have an ``actor``; terminals do not."""

    id: int
    history: tuple[Action, ...]
    contributions: tuple[float, float]
    actor: int | None
    actions: tuple[Action, ...] = ()
    children: tuple[int, ...] = ()
    terminal: TerminalKind | None = None
    folder: int | None = None

    @property
    def is_terminal(self) -> bool:
        return self.actor is None

    @property
    def history_labels(self) -> tuple[str, ...]:
        return tuple(a.label for a in self.history)


@dataclass(frozen=True)
class Infoset:
    """What a player knows when acting: their own card (None if uninformed) and the betting so far."""

    player: int
    card: int | None
    history: tuple[str, ...]

    def __str__(self) -> str:
        card = "?" if self.card is None else str(self.card)
        return f"P{self.player} card={card} | {' '.join(self.history) or '(start)'}"


class GameTree:
    """The full betting tree of one game, with information sets and payoff tables."""

    def __init__(self, config: GameConfig) -> None:
        self.config = config
        self.nodes: list[Node] = []
        self._expand(history=(), bets=(0.0, 0.0), actor=config.first_actor, raises=0)

        self.decision_ids: list[int] = [n.id for n in self.nodes if not n.is_terminal]
        self.terminal_ids: list[int] = [n.id for n in self.nodes if n.is_terminal]

        # payoffs[k, i, j] = payoff to player 0 at terminal_ids[k] when player 0 holds
        # rank i and player 1 holds rank j. Player 1's payoff is the negative.
        self.payoffs: FloatArray = np.stack(
            [self._payoff_p0(self.nodes[t]) for t in self.terminal_ids]
        )

        deal = config.deal_distribution()
        self.card_marginals: tuple[FloatArray, FloatArray] = (deal.sum(axis=1), deal.sum(axis=0))

        self.infosets: list[Infoset] = []
        self.infoset_index: dict[Infoset, int] = {}
        self.infoset_node: dict[Infoset, int] = {}
        self._index_infosets()

    @property
    def root(self) -> Node:
        return self.nodes[0]

    def infoset_of(self, node_id: int, card: int) -> Infoset:
        """The information set of the player acting at ``node_id`` when holding ``card``."""
        node = self.nodes[node_id]
        if node.actor is None:
            raise ValueError(f"node {node_id} is terminal")
        known = card if self.config.informed[node.actor] else None
        return Infoset(node.actor, known, node.history_labels)

    def infosets_of(self, player: int) -> list[Infoset]:
        return [s for s in self.infosets if s.player == player]

    def actions_at(self, infoset: Infoset) -> tuple[Action, ...]:
        return self.nodes[self.infoset_node[infoset]].actions

    def _pot(self, bets: tuple[float, float]) -> float:
        return self.config.initial_pot + bets[0] + bets[1]

    def _totals(self, bets: tuple[float, float]) -> tuple[float, float]:
        a0, a1 = self.config.antes
        return (a0 + bets[0], a1 + bets[1])

    def _terminal(
        self,
        history: tuple[Action, ...],
        bets: tuple[float, float],
        kind: TerminalKind,
        folder: int | None = None,
    ) -> int:
        node = Node(
            id=len(self.nodes),
            history=history,
            contributions=self._totals(bets),
            actor=None,
            terminal=kind,
            folder=folder,
        )
        self.nodes.append(node)
        return node.id

    def _expand(
        self,
        history: tuple[Action, ...],
        bets: tuple[float, float],
        actor: int,
        raises: int,
    ) -> int:
        cfg = self.config
        node = Node(
            id=len(self.nodes), history=history, contributions=self._totals(bets), actor=actor
        )
        self.nodes.append(node)
        opp = 1 - actor

        actions: list[Action]
        if bets[opp] > bets[actor]:
            actions = [Action("fold"), Action("call")]
            if raises < cfg.max_raises:
                actions += [Action("raise", s) for s in cfg.effective_raise_sizes]
        else:
            actions = [Action("check")] + [Action("bet", s) for s in cfg.bet_sizes]

        children: list[int] = []
        for a in actions:
            h = (*history, a)
            if a.kind == "fold":
                child = self._terminal(h, bets, "fold", folder=actor)
            elif a.kind == "call":
                child = self._terminal(h, _with(bets, actor, bets[opp]), "showdown")
            elif a.kind == "check":
                if history and history[-1].kind == "check":
                    child = self._terminal(h, bets, "showdown")
                else:
                    child = self._expand(h, bets, opp, raises)
            elif a.kind == "bet":
                assert a.size is not None
                new = _with(bets, actor, bets[actor] + a.size * self._pot(bets))
                child = self._expand(h, new, opp, raises)
            else:  # raise: match, then add size * (pot after matching)
                assert a.size is not None
                matched = _with(bets, actor, bets[opp])
                new = _with(matched, actor, matched[actor] + a.size * self._pot(matched))
                child = self._expand(h, new, opp, raises + 1)
            children.append(child)

        node.actions = tuple(actions)
        node.children = tuple(children)
        return node.id

    def _payoff_p0(self, node: Node) -> FloatArray:
        n = self.config.num_ranks
        t0, t1 = node.contributions
        if node.terminal == "fold":
            value = t1 if node.folder == 1 else -t0
            return np.full((n, n), value, dtype=np.float64)
        ranks = np.arange(n)
        diff = ranks[:, None] - ranks[None, :]
        if self.config.payoff == "low":
            diff = -diff
        # Winner takes the whole pot; a tie splits it.
        return np.where(diff > 0, t1, np.where(diff < 0, -t0, (t1 - t0) / 2)).astype(np.float64)

    def _index_infosets(self) -> None:
        for nid in self.decision_ids:
            actor = self.nodes[nid].actor
            assert actor is not None
            if self.config.informed[actor]:
                cards = [int(c) for c in np.flatnonzero(self.card_marginals[actor] > 0)]
            else:
                cards = [0]  # any card: an uninformed player's infoset ignores it
            for card in cards:
                infoset = self.infoset_of(nid, card)
                if infoset not in self.infoset_index:
                    self.infoset_index[infoset] = len(self.infosets)
                    self.infosets.append(infoset)
                    self.infoset_node[infoset] = nid


def _with(bets: tuple[float, float], player: int, value: float) -> tuple[float, float]:
    return (value, bets[1]) if player == 0 else (bets[0], value)
