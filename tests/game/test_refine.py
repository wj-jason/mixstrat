from pathlib import Path

import numpy as np
import pytest

from mixstrat.game.best_response import exploitability
from mixstrat.game.cfr import solve
from mixstrat.game.config import GameConfig, load_games
from mixstrat.game.refine import infoset_reach, off_path, refine_off_path
from mixstrat.game.tree import GameTree, Infoset

PRESETS = Path(__file__).resolve().parents[2] / "configs" / "games.toml"
K = 2


@pytest.fixture(scope="module")
def presets() -> dict[str, GameConfig]:
    return load_games(PRESETS)


def test_reach_sums_to_one_at_the_root(presets) -> None:
    tree = GameTree(presets["kuhn_classic"])
    reach = infoset_reach(tree, solve(tree, 500).profile)
    assert sum(r for s, r in reach.items() if s.history == ()) == pytest.approx(1.0)


def test_quarter_pot_king_calls(presets) -> None:
    """P0 always bets K at quarter pot, so 'K, check, facing a bet' is off-path; CFR's
    leftover average folds the best card there, the refined profile calls."""
    tree = GameTree(presets["kuhn_quarter_pot"])
    spot = Infoset(0, K, ("check", "bet:0.25"))
    raw = solve(tree, 2000, refine=False).profile
    assert spot in off_path(tree, raw)
    assert raw.at(spot)[0] > 0.05  # folds K
    refined = solve(tree, 2000).profile
    assert np.allclose(refined.at(spot), [0.0, 1.0])


@pytest.mark.parametrize(
    "game", ["kuhn_quarter_pot", "kuhn_pot", "kuhn_overbet", "kuhn13_one_raise"]
)
def test_refining_keeps_the_path_and_exploitability(presets, game) -> None:
    tree = GameTree(presets[game])
    raw = solve(tree, 2000, refine=False).profile
    refined = refine_off_path(tree, raw)
    off = off_path(tree, raw)
    for s in tree.infosets:
        if s not in off:
            assert np.allclose(raw.at(s), refined.at(s))
    pot = tree.config.initial_pot
    assert exploitability(tree, refined).normalized(pot) <= (
        exploitability(tree, raw).normalized(pot) + 1e-6
    )


def test_no_change_when_everything_is_reached(presets) -> None:
    tree = GameTree(presets["kuhn_classic"])
    raw = solve(tree, 2000, refine=False).profile
    assert not off_path(tree, raw)
    assert refine_off_path(tree, raw) is raw
