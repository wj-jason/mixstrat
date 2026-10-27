import tomllib

import numpy as np
import pytest

from mixstrat.train.sweep import SWEEPS, choose_options, expand, train_args


def test_expand_crosses_variants_with_grid() -> None:
    table = {
        "base": {"steps": 10},
        "grid": {"beta": [0.1, 0.2], "seed": [0, 1]},
        "variants": {"A": {"game": ["kuhn_classic"]}, "B": {"game": ["kuhn_classic", "kuhn13"]}},
    }
    runs = expand("s", table)
    assert len(runs) == 8
    assert runs[0].name == "s/A-beta=0.1-seed=0"
    assert runs[0].options == {"steps": 10, "game": ["kuhn_classic"], "beta": 0.1, "seed": 0}
    assert runs[-1].slug == "s__B-beta=0.2-seed=1"


def test_expand_without_variants() -> None:
    runs = expand("s", {"grid": {"window": [1, 10]}})
    assert [r.name for r in runs] == ["s/window=1", "s/window=10"]


def test_train_args() -> None:
    args = train_args({"held_out": ["a", "b"], "beta": 0.2, "no_shuffle": True, "track": False})
    assert args == ["--held-out", "a", "b", "--beta", "0.2", "--no-shuffle"]


def test_configured_sweeps_parse_into_valid_train_args() -> None:
    from mixstrat.game.config import load_games
    from mixstrat.train.common import PRESETS

    presets = load_games(PRESETS)
    with open(SWEEPS, "rb") as f:
        sweeps = tomllib.load(f)
    for name, table in sweeps.items():
        for run in expand(name, table):
            for game in run.options.get("game", []) + run.options.get("held_out", []):
                assert game in presets, (name, game)


def test_sample_items_splits_evenly_across_games() -> None:
    pytest.importorskip("torch")
    from mixstrat.game.cfr import solve
    from mixstrat.game.config import GameConfig
    from mixstrat.game.tree import GameTree
    from mixstrat.train.common import Game, sample_items

    small = GameTree(GameConfig(name="k3"))
    big = GameTree(GameConfig.uniform("k13", 13))
    games = [Game(t, solve(t, 10).profile, t.config.name) for t in (small, big)]
    rng = np.random.default_rng(0)
    items = sample_items(games, 24, rng)
    assert sum(g.name == "k3" for g, _ in items) == 12  # all 12 (its share is 12)
    assert sum(g.name == "k13" for g, _ in items) == 12
    assert len(sample_items(games, 0, rng)) == len(small.infosets) + len(big.infosets)


def test_choose_options_prefers_low_and_stable() -> None:
    runs = [
        ({"beta": 0.02, "window": 1, "seed": 0}, [0.20, 0.07, 0.17, 0.08, 0.16, 0.07]),  # cycling
        ({"beta": 0.02, "window": 1, "seed": 1}, [0.20, 0.09, 0.15, 0.08, 0.14, 0.09]),
        ({"beta": 0.2, "window": 10, "seed": 0}, [0.20, 0.12, 0.10, 0.09, 0.09, 0.09]),  # settles
        ({"beta": 0.2, "window": 10, "seed": 1}, [0.20, 0.11, 0.10, 0.10, 0.09, 0.09]),
    ]
    chosen, scores = choose_options(runs, ["beta", "window"], last=5)
    assert chosen == {"beta": 0.2, "window": 10}
    assert scores[(0.2, 10)] == pytest.approx(np.mean([0.098, 0.098]))
