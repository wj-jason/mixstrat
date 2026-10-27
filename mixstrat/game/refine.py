"""Sensible play at the infosets an equilibrium never reaches.

CFR returns the average strategy, and each iteration counts in proportion to
how often the player reaches the infoset. Once a player stops going somewhere
(in quarter-pot Kuhn player 0 always bets K, so never checks it), that
infoset's average freezes at its early, half-trained value: player 0 holding K
and facing a bet folds 14% of the time. No payoff on the equilibrium path
depends on it, so the profile is still an equilibrium, but the leftover is a
nonsense distillation target and a poor opponent after a mistake.

``refine_off_path`` replaces the strategy at every infoset the profile (almost)
never reaches by a best response there: the highest-value actions given
that play reached the infoset, continuing under the refined profile. If the
opponent's strategy never reaches the infoset either, the values come from a
slightly trembled opponent (each action at least ``tremble``), i.e. a belief
over their cards as if they got there by mistake. Deeper infosets are refined
first, so shallower ones see the refined continuation. On-path strategies,
and so every on-path payoff and the exploitability, are unchanged.
"""

from __future__ import annotations

import numpy as np

from mixstrat.game.strategy import Profile
from mixstrat.game.tree import FloatArray, GameTree, Infoset
from mixstrat.game.values import all_counterfactual_values

# CFR's average strategy is never exactly 0, so neither is the reach of an
# abandoned infoset; it decays with iterations instead (1e-6 to 1e-8 at 2000
# CFR+ iterations on the 3-card presets, against >= 1e-2 for every infoset on
# the path).
REACH_EPS = 1e-5


def infoset_reach(tree: GameTree, profile: Profile) -> dict[Infoset, float]:
    """Probability that play under ``profile`` (chance and both players) reaches each infoset."""
    deal = tree.config.deal_distribution()  # (own card of player 0, of player 1)
    joint: dict[int, FloatArray] = {}  # node -> (n, n) reach per deal

    def visit(nid: int, r0: FloatArray, r1: FloatArray) -> None:
        node = tree.nodes[nid]
        if node.is_terminal:
            return
        joint[nid] = deal * np.outer(r0, r1)
        sigma = profile.probs[nid]  # (n or 1, A): rows broadcast over cards
        for a, child in enumerate(node.children):
            if node.actor == 0:
                visit(child, r0 * sigma[:, a], r1)
            else:
                visit(child, r0, r1 * sigma[:, a])

    n = tree.config.num_ranks
    visit(0, np.ones(n), np.ones(n))
    reach = {}
    for s in tree.infosets:
        m = joint[tree.infoset_node[s]]
        per_card = m.sum(axis=1) if s.player == 0 else m.sum(axis=0)
        reach[s] = float(per_card.sum() if s.card is None else per_card[s.card])
    return reach


def off_path(tree: GameTree, profile: Profile, eps: float = REACH_EPS) -> set[Infoset]:
    """Infosets that play under ``profile`` never reaches."""
    return {s for s, r in infoset_reach(tree, profile).items() if r <= eps}


def refine_off_path(
    tree: GameTree, profile: Profile, *, tremble: float = 1e-6, tol: float = 1e-5
) -> Profile:
    """``profile`` with every off-path infoset replaced by a best response (see module doc).
    Among actions within ``tol`` chips of the best, the original mix is kept."""
    off = off_path(tree, profile)
    if not off:
        return profile
    table = profile.to_infosets()
    for depth in sorted({len(s.history) for s in off}, reverse=True):
        current = Profile.from_infosets(tree, table)
        values = all_counterfactual_values(tree, current)
        trembled = Profile.from_infosets(
            tree, {s: (1 - tremble) * p + tremble / len(p) for s, p in table.items()}
        )
        values_trembled = all_counterfactual_values(tree, trembled)
        for s in off:
            if len(s.history) != depth:
                continue
            v = values[s] if values[s].reach > REACH_EPS else values_trembled[s]
            best = v.q >= v.q.max() - tol
            kept = np.where(best, table[s], 0.0)
            table[s] = kept / kept.sum() if kept.sum() > 0 else best / best.sum()
    return Profile.from_infosets(tree, table)
