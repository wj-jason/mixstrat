"""Strategy plots: an LLM's action probabilities by card, next to the equilibrium's."""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from mixstrat.game.strategy import Profile
from mixstrat.game.tree import GameTree
from mixstrat.llm.report import node_cards

# Categorical slots in fixed order (one per action position at a node).
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


def strategy_figure(tree: GameTree, llm: Profile, equilibrium: Profile) -> Figure:
    """One panel per decision node: P(action | card), LLM solid, equilibrium dashed."""
    ids = tree.decision_ids
    cols = min(2, len(ids))
    rows = math.ceil(len(ids) / cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(6.4 * cols, 3.2 * rows), squeeze=False, facecolor=SURFACE
    )
    for ax in axes.flat[len(ids) :]:
        ax.set_visible(False)

    for ax, nid in zip(axes.flat, ids, strict=False):
        node = tree.nodes[nid]
        assert node.actor is not None
        cards, labels = node_cards(tree, nid)
        x = np.arange(len(cards))
        for a, action in enumerate(node.actions):
            color = SERIES[a % len(SERIES)]
            ax.plot(x, equilibrium.probs[nid][cards, a], color=color, lw=1.5, ls="--", alpha=0.8)
            ax.plot(
                x, llm.probs[nid][cards, a], color=color, lw=2, marker="o", ms=5, label=action.label
            )

        history = " ".join(node.history_labels) or "(start)"
        ax.set_title(f"Player {node.actor}  |  {history}", loc="left", fontsize=10, color=TEXT)
        ax.set_facecolor(SURFACE)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks(x, labels)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.legend(
            loc="lower right",
            bbox_to_anchor=(1.0, 1.0),
            fontsize=8,
            frameon=False,
            labelcolor=TEXT_SECONDARY,
            ncols=4,
            borderaxespad=0.2,
        )

    style = [
        Line2D([], [], color=TEXT_SECONDARY, lw=2, marker="o", ms=5, label="LLM"),
        Line2D([], [], color=TEXT_SECONDARY, lw=1.5, ls="--", label="equilibrium (CFR+)"),
    ]
    fig.legend(
        handles=style, loc="upper right", ncols=2, frameon=False, fontsize=9, labelcolor=TEXT
    )
    fig.suptitle(
        f"{tree.config.name}: P(action | card)", x=0.01, ha="left", fontsize=12, color=TEXT
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig
