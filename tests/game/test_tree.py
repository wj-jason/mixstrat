from pathlib import Path

import numpy as np
import pytest

from mixstrat.game.config import GameConfig, load_games
from mixstrat.game.tree import GameTree

PRESETS = Path(__file__).resolve().parents[2] / "configs" / "games.toml"


def histories(tree: GameTree, terminal: bool) -> set[str]:
    return {" ".join(n.history_labels) for n in tree.nodes if n.is_terminal == terminal}


@pytest.fixture
def kuhn() -> GameTree:
    return GameTree(GameConfig(name="kuhn"))


def test_kuhn_structure(kuhn: GameTree) -> None:
    assert histories(kuhn, terminal=False) == {"", "check", "bet:0.5", "check bet:0.5"}
    assert histories(kuhn, terminal=True) == {
        "check check",
        "bet:0.5 fold",
        "bet:0.5 call",
        "check bet:0.5 fold",
        "check bet:0.5 call",
    }
    assert kuhn.root.actor == 0


def test_kuhn_payoffs(kuhn: GameTree) -> None:
    sign = np.sign(np.subtract.outer(np.arange(3), np.arange(3)))
    expected = {
        "check check": sign * 1.0,
        "bet:0.5 call": sign * 2.0,
        "bet:0.5 fold": np.full((3, 3), 1.0),
        "check bet:0.5 fold": np.full((3, 3), -1.0),
        "check bet:0.5 call": sign * 2.0,
    }
    for k, tid in enumerate(kuhn.terminal_ids):
        label = " ".join(kuhn.nodes[tid].history_labels)
        mask = ~np.eye(3, dtype=bool)  # equal cards can't be dealt in classic Kuhn
        assert np.array_equal(kuhn.payoffs[k][mask], expected[label][mask]), label


def test_kuhn_infosets(kuhn: GameTree) -> None:
    # 3 cards x 2 betting histories for each player.
    assert len(kuhn.infosets_of(0)) == 6
    assert len(kuhn.infosets_of(1)) == 6
    for s in kuhn.infosets:
        assert len(kuhn.actions_at(s)) == 2


def test_one_raise_structure() -> None:
    tree = GameTree(GameConfig(name="r", max_raises=1, bet_sizes=(1.0,)))
    assert len(tree.decision_ids) == 6
    assert len(tree.terminal_ids) == 9
    # antes 1+1, pot-size bet: bettor puts in 2. Pot-size raise: match to 2 (pot 6), add 6.
    node = next(n for n in tree.nodes if n.history_labels == ("bet:1", "raise:1"))
    assert node.contributions == (3.0, 9.0)


def test_two_bet_sizes() -> None:
    tree = GameTree(GameConfig(name="two", bet_sizes=(0.5, 2.0)))
    assert [a.label for a in tree.root.actions] == ["check", "bet:0.5", "bet:2"]


def test_first_actor_swapped() -> None:
    tree = GameTree(GameConfig(name="swap", first_actor=1))
    assert tree.root.actor == 1
    # Player 0 folding to player 1's bet loses player 0's ante.
    k = tree.terminal_ids.index(
        next(n.id for n in tree.nodes if n.history_labels == ("bet:0.5", "fold"))
    )
    assert np.all(tree.payoffs[k] == -1.0)


def test_lowball_flips_showdown() -> None:
    high = GameTree(GameConfig(name="h"))
    low = GameTree(GameConfig(name="l", payoff="low"))
    k = high.terminal_ids.index(
        next(n.id for n in high.nodes if n.history_labels == ("check", "check"))
    )
    assert np.array_equal(low.payoffs[k], -high.payoffs[k])


def test_ties_split_the_pot() -> None:
    tree = GameTree(GameConfig.uniform("dup", num_ranks=3, copies=2))
    for k in range(len(tree.terminal_ids)):
        if tree.nodes[tree.terminal_ids[k]].terminal == "showdown":
            assert np.all(np.diag(tree.payoffs[k]) == 0.0)


def test_asymmetric_antes_are_not_a_bet() -> None:
    tree = GameTree(GameConfig(name="asym", antes=(0.5, 1.0)))
    assert [a.kind for a in tree.root.actions] == ["check", "bet"]
    k = tree.terminal_ids.index(
        next(n.id for n in tree.nodes if n.history_labels == ("check", "check"))
    )
    assert tree.payoffs[k][2, 0] == 1.0  # player 0 wins player 1's ante
    assert tree.payoffs[k][0, 2] == -0.5  # player 0 loses their own ante


def test_uninformed_player_has_one_infoset_per_history() -> None:
    tree = GameTree(GameConfig.uniform("clair", num_ranks=13, informed=(True, False)))
    assert len(tree.infosets_of(0)) == 13 * 2
    assert len(tree.infosets_of(1)) == 2
    assert all(s.card is None for s in tree.infosets_of(1))


def test_all_presets_build() -> None:
    for name, cfg in load_games(PRESETS).items():
        tree = GameTree(cfg)
        assert tree.payoffs.shape == (len(tree.terminal_ids), cfg.num_ranks, cfg.num_ranks), name
        # Every decision node has at least two actions, and children line up with actions.
        for nid in tree.decision_ids:
            node = tree.nodes[nid]
            assert len(node.actions) == len(node.children) >= 2
