"""CFR+ solver for a ``GameTree``.

Counterfactual regret minimization, CFR+ variant:
- regret matching+ (cumulative regrets floored at zero),
- alternating updates (player 0, then player 1, each iteration),
- linear averaging (iteration t's strategy weighted by t).

The *average* strategy converges to a Nash equilibrium; the current strategy
need not. Each traversal handles all card pairs at once: values are vectors
over the traverser's cards, and reach probabilities are vectors over cards.

For a player who doesn't see their own card, regrets and averages are kept for
a single row (one infoset per node) and summed over cards.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from mixstrat.game.best_response import Exploitability, exploitability, oriented
from mixstrat.game.strategy import Profile
from mixstrat.game.tree import FloatArray, GameTree


@dataclass
class CFRResult:
    profile: Profile  # average strategy: the equilibrium approximation
    iterations: int
    history: list[tuple[int, Exploitability]] = field(default_factory=list)

    @property
    def final(self) -> Exploitability:
        return exploitability(self.profile.tree, self.profile)


class CFRPlus:
    """Stateful CFR+ solver; call ``run`` repeatedly to keep improving."""

    def __init__(self, tree: GameTree) -> None:
        self.tree = tree
        self.n = tree.config.num_ranks
        self.iteration = 0
        self.regrets: dict[int, FloatArray] = {}
        self.avg: dict[int, FloatArray] = {}
        for nid in tree.decision_ids:
            node = tree.nodes[nid]
            assert node.actor is not None
            rows = self.n if tree.config.informed[node.actor] else 1
            self.regrets[nid] = np.zeros((rows, len(node.actions)))
            self.avg[nid] = np.zeros((rows, len(node.actions)))
        self._oriented = (oriented(tree, 0), oriented(tree, 1))
        self._terminal_k = {nid: k for k, nid in enumerate(tree.terminal_ids)}

    # ------------------------------------------------------------ strategies

    def _expand(self, rows: FloatArray) -> FloatArray:
        """Broadcast a single uninformed row to one row per card."""
        return rows if rows.shape[0] == self.n else np.repeat(rows, self.n, axis=0)

    @staticmethod
    def _normalize(x: FloatArray) -> FloatArray:
        total = x.sum(axis=1, keepdims=True)
        uniform = np.full_like(x, 1.0 / x.shape[1])
        return np.where(total > 0, x / np.where(total > 0, total, 1.0), uniform)

    def current_strategy(self, nid: int) -> FloatArray:
        return self._expand(self._normalize(np.maximum(self.regrets[nid], 0.0)))

    def average_profile(self) -> Profile:
        return Profile(
            self.tree,
            {nid: self._expand(self._normalize(a)) for nid, a in self.avg.items()},
        )

    # ------------------------------------------------------------ iteration

    def _traverse(self, traverser: int) -> None:
        tree, n, t = self.tree, self.n, self.iteration
        deal, payoffs = self._oriented[traverser]
        weighted = deal[None, :, :] * payoffs
        sigma = {nid: self.current_strategy(nid) for nid in tree.decision_ids}

        def visit(nid: int, own_reach: FloatArray, opp_reach: FloatArray) -> FloatArray:
            node = tree.nodes[nid]
            if node.is_terminal:
                return weighted[self._terminal_k[nid]] @ opp_reach
            s = sigma[nid]
            if node.actor != traverser:
                return sum(
                    (
                        visit(child, own_reach, opp_reach * s[:, a])
                        for a, child in enumerate(node.children)
                    ),
                    start=np.zeros(n),
                )
            child_values = np.stack(
                [
                    visit(child, own_reach * s[:, a], opp_reach)
                    for a, child in enumerate(node.children)
                ]
            )  # (A, n)
            value = np.einsum("na,an->n", s, child_values)
            instant = child_values.T - value[:, None]  # (n, A) regrets this iteration
            if self.regrets[nid].shape[0] == n:
                self.regrets[nid] = np.maximum(self.regrets[nid] + instant, 0.0)
                self.avg[nid] += t * own_reach[:, None] * s
            else:
                self.regrets[nid] = np.maximum(
                    self.regrets[nid] + instant.sum(axis=0, keepdims=True), 0.0
                )
                # An uninformed player's own reach doesn't depend on their card.
                self.avg[nid] += t * own_reach[0] * s[:1]
            return value

        visit(0, np.ones(n), np.ones(n))

    def run(
        self,
        iterations: int,
        log_every: int = 0,
        callback: Callable[[int, Exploitability], None] | None = None,
    ) -> CFRResult:
        history: list[tuple[int, Exploitability]] = []
        for _ in range(iterations):
            self.iteration += 1
            self._traverse(0)
            self._traverse(1)
            if log_every and self.iteration % log_every == 0:
                report = exploitability(self.tree, self.average_profile())
                history.append((self.iteration, report))
                if callback is not None:
                    callback(self.iteration, report)
        return CFRResult(self.average_profile(), self.iteration, history)


def solve(tree: GameTree, iterations: int = 2000, log_every: int = 0) -> CFRResult:
    """Approximate a Nash equilibrium of ``tree`` with CFR+."""
    return CFRPlus(tree).run(iterations, log_every=log_every)
