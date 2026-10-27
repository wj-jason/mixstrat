"""Counterfactual action values: what each action is worth at each infoset.

For player ``i`` at infoset ``I`` (card ``c``, node ``n``) under profile
``sigma``, the conditional value of action ``a`` is

    q(I, a) = sum_o P(c, o) reach_opp(n, o) V(n_a | c, o)  /  sum_o P(c, o) reach_opp(n, o)

the chips player ``i`` expects from taking ``a`` given that they hold ``c`` and
play reached ``n``, with everything afterwards (both players) following
``sigma``. The denominator is the counterfactual reach: how likely the
opponent's cards and actions make reaching ``n``, ignoring player ``i``'s own
choices. These are the values CFR's regrets are built from, and the rewards for
value-based post-training (see docs/post_training_design.md).

Computations are vectorized over cards like ``best_response``: one traversal
per player handles every deal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mixstrat.game.best_response import oriented
from mixstrat.game.strategy import Profile
from mixstrat.game.tree import FloatArray, GameTree, Infoset


@dataclass(frozen=True)
class ActionValues:
    q: FloatArray  # conditional value of each action at the infoset, in chips
    reach: float  # counterfactual reach probability; q is undefined when this is 0


def counterfactual_values(
    tree: GameTree, profile: Profile, player: int
) -> dict[Infoset, ActionValues]:
    """``ActionValues`` for every infoset of ``player`` under ``profile``."""
    deal, payoffs = oriented(tree, player)
    weighted = deal[None, :, :] * payoffs  # (terminal, own, opp)
    terminal_k = {nid: k for k, nid in enumerate(tree.terminal_ids)}
    n = tree.config.num_ranks
    q_cf: dict[int, FloatArray] = {}  # node -> (actions, own cards), weighted by deal and reach
    reach_cf: dict[int, FloatArray] = {}  # node -> (own cards,)

    def visit(nid: int, opp_reach: FloatArray) -> FloatArray:
        """Counterfactual value to ``player`` for each of their cards."""
        node = tree.nodes[nid]
        if node.is_terminal:
            return weighted[terminal_k[nid]] @ opp_reach
        sigma = profile.probs[nid]
        if node.actor != player:
            return sum(
                (visit(child, opp_reach * sigma[:, a]) for a, child in enumerate(node.children)),
                start=np.zeros(n),
            )
        child_values = np.stack([visit(child, opp_reach) for child in node.children])  # (A, n)
        q_cf[nid] = child_values
        reach_cf[nid] = deal @ opp_reach
        return np.einsum("na,an->n", sigma, child_values)

    visit(0, np.ones(n))

    values = {}
    for infoset in tree.infosets_of(player):
        nid = tree.infoset_node[infoset]
        if infoset.card is None:  # uninformed: one infoset covers every card
            q, reach = q_cf[nid].sum(axis=1), float(reach_cf[nid].sum())
        else:
            q, reach = q_cf[nid][:, infoset.card], float(reach_cf[nid][infoset.card])
        values[infoset] = ActionValues(q / reach if reach > 0 else np.zeros_like(q), reach)
    return values


def all_counterfactual_values(tree: GameTree, profile: Profile) -> dict[Infoset, ActionValues]:
    """``ActionValues`` for every infoset of both players under one profile."""
    return counterfactual_values(tree, profile, 0) | counterfactual_values(tree, profile, 1)
