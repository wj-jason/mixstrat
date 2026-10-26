"""Best responses and exploitability.

A best response is the strategy that wins the most against a fixed opponent.
Exploitability measures how far a profile is from Nash equilibrium: the average,
over the two seats, of how much a best-responding opponent gains beyond the
profile's own value. It is 0 exactly at a Nash equilibrium.

Computations are "oriented": for the player we compute values for, payoffs and
deal probabilities are arranged as (own card, opponent card), so one code path
serves both players.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mixstrat.game.strategy import Profile, expected_value
from mixstrat.game.tree import FloatArray, GameTree


def oriented(tree: GameTree, player: int) -> tuple[FloatArray, FloatArray]:
    """Deal distribution and per-terminal payoffs from ``player``'s point of view.

    Returns ``(deal[own, opp], payoffs[k, own, opp])``.
    """
    deal = tree.config.deal_distribution()
    if player == 0:
        return deal, tree.payoffs
    return deal.T, -np.transpose(tree.payoffs, (0, 2, 1))


def best_response(tree: GameTree, profile: Profile, player: int) -> tuple[Profile, float]:
    """A pure best response for ``player`` against the other player's strategy in ``profile``.

    Returns the profile with ``player``'s strategy replaced by the best response,
    and the best response's expected payoff to ``player``.
    """
    deal, payoffs = oriented(tree, player)
    weighted = deal[None, :, :] * payoffs  # (k, own, opp)
    terminal_k = {nid: k for k, nid in enumerate(tree.terminal_ids)}
    informed = tree.config.informed[player]
    n = tree.config.num_ranks
    br_probs = {nid: arr.copy() for nid, arr in profile.probs.items()}

    def visit(nid: int, opp_reach: FloatArray) -> FloatArray:
        """Counterfactual value to ``player`` for each of their cards."""
        node = tree.nodes[nid]
        if node.is_terminal:
            return weighted[terminal_k[nid]] @ opp_reach
        if node.actor != player:
            sigma = profile.probs[nid]
            return sum(
                (visit(child, opp_reach * sigma[:, a]) for a, child in enumerate(node.children)),
                start=np.zeros(n),
            )
        child_values = np.stack([visit(child, opp_reach) for child in node.children])  # (A, n)
        choice = np.zeros((n, len(node.children)))
        if informed:
            best = np.argmax(child_values, axis=0)
            choice[np.arange(n), best] = 1.0
            value = child_values[best, np.arange(n)]
        else:
            best_a = int(np.argmax(child_values.sum(axis=1)))
            choice[:, best_a] = 1.0
            value = child_values[best_a]
        br_probs[nid] = choice
        return value

    value = float(visit(0, np.ones(n)).sum())
    return Profile(tree, br_probs), value


@dataclass(frozen=True)
class Exploitability:
    """Exploitability report for one profile (all values in chips per hand)."""

    value: float  # expected payoff to player 0 when both follow the profile
    br_value_p0: float  # best response as player 0 against the profile's player 1
    br_value_p1: float  # best response as player 1 against the profile's player 0
    nash_conv: float  # br_value_p0 + br_value_p1 (sum of both players' gains)
    exploitability: float  # nash_conv / 2

    def normalized(self, pot: float) -> float:
        """Exploitability as a fraction of the starting pot, for comparing variants."""
        return self.exploitability / pot


def exploitability(tree: GameTree, profile: Profile) -> Exploitability:
    _, br0 = best_response(tree, profile, 0)
    _, br1 = best_response(tree, profile, 1)
    nash_conv = br0 + br1  # zero-sum: the profile's own values cancel
    return Exploitability(
        value=expected_value(tree, profile),
        br_value_p0=br0,
        br_value_p1=br1,
        nash_conv=nash_conv,
        exploitability=nash_conv / 2,
    )
