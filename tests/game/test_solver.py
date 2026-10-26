from pathlib import Path

import numpy as np
import pytest

from mixstrat.game.best_response import best_response, exploitability
from mixstrat.game.cfr import solve
from mixstrat.game.config import GameConfig, load_games
from mixstrat.game.strategy import Profile, expected_value
from mixstrat.game.tree import GameTree, Infoset

PRESETS = Path(__file__).resolve().parents[2] / "configs" / "games.toml"
J, Q, K = 0, 1, 2


@pytest.fixture(scope="module")
def kuhn() -> GameTree:
    return GameTree(GameConfig(name="kuhn"))


@pytest.fixture(scope="module")
def kuhn_eq(kuhn: GameTree) -> Profile:
    return solve(kuhn, iterations=3000).profile


def p_aggressive(profile: Profile, player: int, card: int, *history: str) -> float:
    """Probability of the second action (bet or call) at an infoset."""
    return float(profile.at(Infoset(player, card, tuple(history)))[1])


# ---------------------------------------------------------------- profiles


def test_profile_rejects_bad_rows(kuhn: GameTree) -> None:
    probs = Profile.uniform(kuhn).probs
    probs[0] = probs[0] * 2
    with pytest.raises(ValueError):
        Profile(kuhn, probs)


def test_from_infosets_round_trip(kuhn: GameTree) -> None:
    uniform = Profile.uniform(kuhn)
    rebuilt = Profile.from_infosets(kuhn, uniform.to_infosets())
    for nid in kuhn.decision_ids:
        assert np.allclose(rebuilt.probs[nid], uniform.probs[nid])


def always(tree: GameTree, action_index: int) -> Profile:
    return Profile.from_infosets(tree, {s: np.eye(2)[action_index] for s in tree.infosets})


def test_expected_value_symmetric_profiles(kuhn: GameTree) -> None:
    # Always check / always fold: check-check showdown, symmetric -> 0.
    assert expected_value(kuhn, always(kuhn, 0)) == pytest.approx(0.0)
    # Always bet / always call: bet-call showdown, symmetric -> 0.
    assert expected_value(kuhn, always(kuhn, 1)) == pytest.approx(0.0)


# ---------------------------------------------------------------- best response


def test_best_response_beats_or_ties_profile(kuhn: GameTree) -> None:
    uniform = Profile.uniform(kuhn)
    v = expected_value(kuhn, uniform)
    _, br0 = best_response(kuhn, uniform, 0)
    _, br1 = best_response(kuhn, uniform, 1)
    assert br0 >= v - 1e-12
    assert br1 >= -v - 1e-12


def test_exploit_always_call(kuhn: GameTree) -> None:
    # Against a player 1 who never bets and always calls, player 0's best response
    # bets K, checks J (never bluffs), and its value is computable by hand:
    # P0 K: bet, gets called, wins 2 always -> +2 ; P0 Q: bet wins 2 vs J, loses 2 vs K -> 0,
    # check -> showdown +-1 -> 0 ; P0 J: check -> -1 ; average = (2 + 0 - 1) / 3.
    station = always(kuhn, 1)
    probs = station.probs.copy()
    check_id = kuhn.infoset_node[Infoset(1, J, ("check",))]
    probs[check_id] = np.tile([1.0, 0.0], (3, 1))  # player 1 never bets
    station = Profile(kuhn, probs)
    _, value = best_response(kuhn, station, 0)
    assert value == pytest.approx(1 / 3)


def test_exploitability_nonnegative(kuhn: GameTree) -> None:
    report = exploitability(kuhn, Profile.uniform(kuhn))
    assert report.exploitability > 0.1


# ---------------------------------------------------------------- CFR on classic Kuhn


def test_kuhn_value_and_exploitability(kuhn: GameTree, kuhn_eq: Profile) -> None:
    report = exploitability(kuhn, kuhn_eq)
    assert report.value == pytest.approx(-1 / 18, abs=2e-3)
    assert report.exploitability < 2e-3


def test_kuhn_equilibrium_frequencies(kuhn_eq: Profile) -> None:
    # Player 1 (known unique values).
    assert p_aggressive(kuhn_eq, 1, Q, "bet:0.5") == pytest.approx(1 / 3, abs=0.03)
    assert p_aggressive(kuhn_eq, 1, J, "check") == pytest.approx(1 / 3, abs=0.03)
    assert p_aggressive(kuhn_eq, 1, K, "check") == pytest.approx(1.0, abs=0.01)
    assert p_aggressive(kuhn_eq, 1, J, "bet:0.5") == pytest.approx(0.0, abs=0.01)
    # Player 0: a one-parameter family with J-bluff alpha in [0, 1/3].
    alpha = p_aggressive(kuhn_eq, 0, J)
    assert -0.01 <= alpha <= 1 / 3 + 0.01
    assert p_aggressive(kuhn_eq, 0, K) == pytest.approx(3 * alpha, abs=0.05)
    assert p_aggressive(kuhn_eq, 0, Q) == pytest.approx(0.0, abs=0.02)
    assert p_aggressive(kuhn_eq, 0, Q, "check", "bet:0.5") == pytest.approx(alpha + 1 / 3, abs=0.05)


# ---------------------------------------------------------------- variants


def test_kuhn13_converges() -> None:
    tree = GameTree(GameConfig.uniform("k13", num_ranks=13))
    report = exploitability(tree, solve(tree, iterations=2000).profile)
    assert report.normalized(tree.config.initial_pot) < 1e-3


@pytest.mark.slow
def test_all_presets_converge() -> None:
    for name, cfg in load_games(PRESETS).items():
        tree = GameTree(cfg)
        report = exploitability(tree, solve(tree, iterations=2000).profile)
        assert report.normalized(cfg.initial_pot) < 5e-3, name
