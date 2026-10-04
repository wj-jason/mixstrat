from pathlib import Path

import numpy as np
import pytest

from mixstrat.game.config import GameConfig, load_games

PRESETS = Path(__file__).resolve().parents[2] / "configs" / "games.toml"


def test_classic_kuhn_defaults() -> None:
    g = GameConfig(name="kuhn")
    assert g.num_ranks == 3
    assert g.initial_pot == 2.0
    assert g.bet_sizes == (0.5,)


def test_shared_deck_no_duplicates() -> None:
    p = GameConfig(name="kuhn").deal_distribution()
    assert p.sum() == pytest.approx(1.0)
    assert np.all(np.diag(p) == 0.0)
    off_diag = p[~np.eye(3, dtype=bool)]
    assert np.allclose(off_diag, 1 / 6)


def test_shared_deck_with_duplicates() -> None:
    # 3 ranks x 2 copies = 6 cards; 6 * 5 = 30 ordered deals.
    p = GameConfig.uniform("dup", num_ranks=3, copies=2).deal_distribution()
    assert p.sum() == pytest.approx(1.0)
    assert np.allclose(np.diag(p), 2 / 30)
    assert p[0, 1] == pytest.approx(4 / 30)


def test_separate_decks_are_independent() -> None:
    g = GameConfig(name="asym", deck=(1, 3), deck_p2=(1, 1))
    p = g.deal_distribution()
    assert np.allclose(p, np.outer([0.25, 0.75], [0.5, 0.5]))


def test_lists_become_tuples() -> None:
    g = GameConfig.from_dict("x", {"num_ranks": 4, "bet_sizes": [0.5, 1.0]})
    assert g.deck == (1, 1, 1, 1)
    assert g.bet_sizes == (0.5, 1.0)
    hash(g)  # configs must be hashable


@pytest.mark.parametrize(
    "data",
    [
        {"bet_sizes": [0.0]},
        {"bet_sizes": [0.5, 0.5]},
        {"antes": [1.0, -1.0]},
        {"deck": [1]},
        {"deck": [1, 1], "deck_p2": [1, 1, 1]},
        {"first_actor": 2},
        {"payoff": "middle"},
        {"num_ranks": 3, "deck": [1, 1, 1]},
        {"not_a_field": 1},
    ],
)
def test_invalid_configs_rejected(data: dict) -> None:
    with pytest.raises(ValueError):
        GameConfig.from_dict("bad", data)


def test_presets_load() -> None:
    games = load_games(PRESETS)
    assert "kuhn_classic" in games
    assert games["kuhn13"].num_ranks == 13
    for g in games.values():
        assert g.deal_distribution().sum() == pytest.approx(1.0)
