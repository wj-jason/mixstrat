from pathlib import Path

import numpy as np

from mixstrat.game.cfr import solve
from mixstrat.game.config import GameConfig, load_games
from mixstrat.game.tree import GameTree
from mixstrat.llm.prompts import build_prompt
from mixstrat.llm.relabel import (
    NAME_POOLS,
    PROBE_NAMES,
    Relabeler,
    map_infoset,
    random_names,
    relabel_config,
    relabeled_tree,
)

PRESETS = Path(__file__).resolve().parents[2] / "configs" / "games.toml"


def test_flipped_view_has_the_same_equilibrium() -> None:
    """A flip moves rank i to n-1-i under the reversed payoff; its strategy must not change."""
    cfg = GameConfig.uniform("k5", 5)
    tree = GameTree(cfg)
    flipped = GameTree(relabel_config(cfg, ("a", "b", "c", "d", "e"), flip=True))
    eq = solve(tree, 3000).profile
    eq_flipped = solve(flipped, 3000).profile
    for s in tree.infosets:
        assert np.allclose(eq.at(s), eq_flipped.at(map_infoset(s, 5, True)), atol=0.03)


def test_flipped_classic_prompt_matches_the_probe_preset() -> None:
    presets = load_games(PRESETS)
    tree = GameTree(presets["kuhn_classic"])
    probe = GameTree(presets["kuhn_lowball_reversed"])
    view = relabeled_tree(tree, ("J", "Q", "K"), flip=True)
    for s in tree.infosets:
        got = build_prompt(view, map_infoset(s, 3, True)).messages
        # Same card name and history; only the description line differs.
        assert got == build_prompt(probe, map_infoset(s, 3, True)).messages


def test_random_names_avoid_probe_labels() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        names = random_names(13, rng)
        assert len(set(names)) == 13 and not PROBE_NAMES & set(names)
    assert all(len(pool) >= 13 for pool in NAME_POOLS.values())


def test_relabeler_rates() -> None:
    tree = GameTree(GameConfig(name="k"))
    s = tree.infosets[0]
    never = Relabeler(0.0, 0.5, np.random.default_rng(0))
    assert never(tree, s) == (tree, s)
    always = Relabeler(1.0, 1.0, np.random.default_rng(0))
    view, mapped = always(tree, s)
    assert view.config.payoff == "low" and mapped.card == 2 - s.card
    assert tree.config.card_names is None  # the original is untouched
