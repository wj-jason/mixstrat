import numpy as np
import pytest

from mixstrat.game.cfr import solve
from mixstrat.game.config import GameConfig
from mixstrat.game.strategy import Profile, expected_value
from mixstrat.game.tree import GameTree, Infoset
from mixstrat.game.values import all_counterfactual_values, counterfactual_values

J, Q, K = 0, 1, 2


@pytest.fixture(scope="module")
def kuhn() -> GameTree:
    return GameTree(GameConfig(name="kuhn"))


def test_values_by_hand_against_always_call(kuhn: GameTree) -> None:
    # Player 1 always checks behind and always calls. Player 0 with K: betting wins
    # the ante plus the 1-chip bet (+2), checking wins the ante at showdown (+1).
    probs = Profile.uniform(kuhn).probs
    for nid in kuhn.decision_ids:
        node = kuhn.nodes[nid]
        if node.actor == 1:
            probs[nid] = np.tile(
                [1.0, 0.0] if node.history_labels == ("check",) else [0.0, 1.0], (3, 1)
            )
    values = counterfactual_values(kuhn, Profile(kuhn, probs), 0)
    root_k = values[Infoset(0, K, ())]
    assert root_k.q == pytest.approx([1.0, 2.0])
    assert root_k.reach == pytest.approx(1 / 3)  # P(holding K); no opponent action before


def test_values_reproduce_expected_value(kuhn: GameTree) -> None:
    profile = Profile.uniform(kuhn)
    values = counterfactual_values(kuhn, profile, 0)
    root_value = sum(
        v.reach * float(profile.at(s) @ v.q) for s, v in values.items() if s.history == ()
    )
    assert root_value == pytest.approx(expected_value(kuhn, profile))


def test_equilibrium_mixes_only_between_equal_values(kuhn: GameTree) -> None:
    eq = solve(kuhn, iterations=5000).profile
    for s, v in all_counterfactual_values(kuhn, eq).items():
        if v.reach < 1e-9:
            continue
        mixed = eq.at(s) > 0.05
        if mixed.sum() > 1:  # indifference: every action played is worth the same
            assert np.ptp(v.q[mixed]) < 0.01, str(s)


def test_uninformed_player_gets_one_value_per_history() -> None:
    tree = GameTree(GameConfig.uniform("clair", 5, informed=(True, False)))
    values = counterfactual_values(tree, Profile.uniform(tree), 1)
    assert all(s.card is None for s in values)
    assert len(values) == 2
