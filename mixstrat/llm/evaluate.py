"""Score an LLM policy against the game's equilibrium."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mixstrat.game.best_response import Exploitability, exploitability
from mixstrat.game.refine import off_path
from mixstrat.game.strategy import Profile, expected_value
from mixstrat.game.tree import GameTree
from mixstrat.llm.policy import PolicyResult
from mixstrat.llm.prompts import card_names


@dataclass(frozen=True)
class Evaluation:
    llm: Exploitability
    uniform: Exploitability
    equilibrium: Exploitability
    seat_ev: tuple[float, float]  # LLM's EV in each seat against the equilibrium opponent

    @property
    def seat_loss(self) -> tuple[float, float]:
        """Chips per hand the LLM gives up in each seat versus playing the equilibrium."""
        v = self.equilibrium.value
        return (v - self.seat_ev[0], -v - self.seat_ev[1])

    def metrics(self, pot: float) -> dict[str, float]:
        return {
            "exploitability": self.llm.exploitability,
            "exploitability_norm": self.llm.normalized(pot),
            "nash_conv": self.llm.nash_conv,
            "br_value_p0": self.llm.br_value_p0,
            "br_value_p1": self.llm.br_value_p1,
            "self_play_value": self.llm.value,
            "seat0_ev_vs_eq": self.seat_ev[0],
            "seat1_ev_vs_eq": self.seat_ev[1],
            "seat0_loss_vs_eq": self.seat_loss[0],
            "seat1_loss_vs_eq": self.seat_loss[1],
            "game_value": self.equilibrium.value,
            "uniform_exploitability_norm": self.uniform.normalized(pot),
            "eq_exploitability_norm": self.equilibrium.normalized(pot),
        }


def evaluate(tree: GameTree, llm: Profile, equilibrium: Profile) -> Evaluation:
    seat0 = expected_value(tree, Profile.combine(llm, equilibrium))
    seat1 = -expected_value(tree, Profile.combine(equilibrium, llm))
    return Evaluation(
        llm=exploitability(tree, llm),
        uniform=exploitability(tree, Profile.uniform(tree)),
        equilibrium=exploitability(tree, equilibrium),
        seat_ev=(seat0, seat1),
    )


def infoset_rows(
    tree: GameTree, result: PolicyResult, equilibrium: Profile
) -> list[dict[str, Any]]:
    """One row per infoset: the LLM's and the equilibrium's action distributions.

    ``tv`` is the total-variation distance between the two. Equilibria needn't be
    unique, so a large ``tv`` flags a difference worth a look, not necessarily a mistake.
    """
    names = card_names(tree.config)
    off = off_path(tree, equilibrium)
    rows = []
    for q in result.queries:
        s = q.infoset
        eq = equilibrium.at(s)
        rows.append(
            {
                "player": s.player,
                "card": "?" if s.card is None else names[s.card],
                "history": " ".join(s.history) or "(start)",
                "actions": " | ".join(a.label for a in tree.actions_at(s)),
                "llm": [round(float(p), 4) for p in q.probs],
                "equilibrium": [round(float(p), 4) for p in eq],
                "tv": round(0.5 * float(np.abs(q.probs - eq).sum()), 4),
                "legal_mass": round(q.legal_mass, 4),
                "off_path": s in off,
            }
        )
    return rows
