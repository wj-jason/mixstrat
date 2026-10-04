"""Strategy profiles: what both players do at every decision point.

A ``Profile`` stores, for each decision node, an ``(n_ranks, n_actions)`` array:
row ``c`` is the acting player's action distribution when holding rank ``c``.
For a player who doesn't see their own card, every row is the same.

Because each betting history leads to exactly one node, a node plus the acting
player's card identifies an information set, so this layout covers every
infoset in the tree. ``from_infosets`` / ``to_infosets`` convert to and from
infoset-keyed tables, which is how strategies from other sources (an LLM, a
rule-based bot) come in.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from mixstrat.game.tree import FloatArray, GameTree, Infoset

_TOL = 1e-6


class Profile:
    """Behaviour strategies for both players over one ``GameTree``."""

    def __init__(self, tree: GameTree, probs: Mapping[int, FloatArray]) -> None:
        self.tree = tree
        self.probs: dict[int, FloatArray] = {}
        n = tree.config.num_ranks
        for nid in tree.decision_ids:
            if nid not in probs:
                raise ValueError(f"no strategy for decision node {nid}")
            node = tree.nodes[nid]
            p = np.asarray(probs[nid], dtype=np.float64)
            if p.shape != (n, len(node.actions)):
                raise ValueError(
                    f"node {nid}: expected shape {(n, len(node.actions))}, got {p.shape}"
                )
            if np.any(p < -_TOL) or not np.allclose(p.sum(axis=1), 1.0, atol=_TOL):
                raise ValueError(f"node {nid}: rows must be probability distributions")
            assert node.actor is not None
            if not tree.config.informed[node.actor] and not np.allclose(p, p[0], atol=_TOL):
                raise ValueError(f"node {nid}: uninformed player's rows must be identical")
            self.probs[nid] = np.clip(p, 0.0, None)

    # ------------------------------------------------------------ construction

    @classmethod
    def uniform(cls, tree: GameTree) -> Profile:
        n = tree.config.num_ranks
        return cls(
            tree,
            {
                nid: np.full((n, len(tree.nodes[nid].actions)), 1.0 / len(tree.nodes[nid].actions))
                for nid in tree.decision_ids
            },
        )

    @classmethod
    def from_infosets(
        cls,
        tree: GameTree,
        table: Mapping[Infoset, Sequence[float] | FloatArray],
        fill_uniform: bool = False,
    ) -> Profile:
        """Build from per-infoset action distributions.

        Infosets missing from ``table`` raise an error unless ``fill_uniform`` is
        set. Cards that can never be dealt to a player always get a uniform row.
        """
        probs = {nid: arr.copy() for nid, arr in cls.uniform(tree).probs.items()}
        for infoset in tree.infosets:
            nid = tree.infoset_node[infoset]
            if infoset in table:
                row = np.asarray(table[infoset], dtype=np.float64)
            elif fill_uniform:
                continue
            else:
                raise ValueError(f"no strategy for infoset {infoset}")
            if infoset.card is None:
                probs[nid][:] = row
            else:
                probs[nid][infoset.card] = row
        return cls(tree, probs)

    @staticmethod
    def combine(p0_from: Profile, p1_from: Profile) -> Profile:
        """Player 0 plays ``p0_from``'s strategy; player 1 plays ``p1_from``'s."""
        tree = p0_from.tree
        if p1_from.tree is not tree:
            raise ValueError("profiles must share the same GameTree")
        probs = {}
        for nid in tree.decision_ids:
            source = p0_from if tree.nodes[nid].actor == 0 else p1_from
            probs[nid] = source.probs[nid]
        return Profile(tree, probs)

    # ------------------------------------------------------------ access

    def at(self, infoset: Infoset) -> FloatArray:
        row = 0 if infoset.card is None else infoset.card
        return self.probs[self.tree.infoset_node[infoset]][row]

    def to_infosets(self) -> dict[Infoset, FloatArray]:
        return {s: self.at(s) for s in self.tree.infosets}


def node_values(tree: GameTree, profile: Profile) -> dict[int, FloatArray]:
    """For every node, the ``(n, n)`` matrix of expected payoff to player 0 per deal."""
    values: dict[int, FloatArray] = {}
    terminal_k = {nid: k for k, nid in enumerate(tree.terminal_ids)}

    def visit(nid: int) -> FloatArray:
        node = tree.nodes[nid]
        if node.is_terminal:
            v = tree.payoffs[terminal_k[nid]]
        else:
            sigma = profile.probs[nid]
            v = np.zeros_like(tree.payoffs[0])
            for a, child in enumerate(node.children):
                cv = visit(child)
                weight = sigma[:, a][:, None] if node.actor == 0 else sigma[:, a][None, :]
                v = v + weight * cv
        values[nid] = v
        return v

    visit(0)
    return values


def expected_value(tree: GameTree, profile: Profile) -> float:
    """Expected payoff to player 0 (player 1 gets the negative)."""
    deal = tree.config.deal_distribution()
    return float(np.sum(deal * node_values(tree, profile)[0]))
