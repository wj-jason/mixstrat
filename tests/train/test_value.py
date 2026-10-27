from collections import deque

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixstrat.game.cfr import solve  # noqa: E402
from mixstrat.game.config import GameConfig  # noqa: E402
from mixstrat.game.strategy import Profile  # noqa: E402
from mixstrat.game.tree import GameTree, Infoset  # noqa: E402
from mixstrat.game.values import ActionValues, counterfactual_values  # noqa: E402
from mixstrat.train.common import Game  # noqa: E402
from mixstrat.train.value import average_values, training_values  # noqa: E402


@pytest.fixture(scope="module")
def kuhn() -> Game:
    tree = GameTree(GameConfig(name="kuhn"))
    return Game(tree, solve(tree, iterations=2000).profile)


def test_values_vs_cfr_put_the_model_in_its_own_seat(kuhn: Game) -> None:
    model = Profile.uniform(kuhn.tree)
    values = training_values(kuhn, model, "cfr")
    p0 = counterfactual_values(kuhn.tree, Profile.combine(model, kuhn.equilibrium), 0)
    p1 = counterfactual_values(kuhn.tree, Profile.combine(kuhn.equilibrium, model), 1)
    for s, v in (p0 | p1).items():
        assert np.allclose(values[s].q, v.q)


def test_self_play_values_use_the_model_everywhere(kuhn: Game) -> None:
    model = Profile.uniform(kuhn.tree)
    values = training_values(kuhn, model, "self")
    expected = counterfactual_values(kuhn.tree, model, 1)
    s = Infoset(1, 0, ("bet:0.5",))
    assert np.allclose(values[s].q, expected[s].q)


def test_average_values_weight_by_reach() -> None:
    s = Infoset(0, 0, ())
    history = deque(
        [
            {s: ActionValues(np.array([1.0, 0.0]), 0.1)},
            {s: ActionValues(np.array([0.0, 1.0]), 0.3)},
        ]
    )
    avg = average_values(history)[s]
    assert avg.q == pytest.approx([0.25, 0.75])
    assert avg.reach == pytest.approx(0.2)
