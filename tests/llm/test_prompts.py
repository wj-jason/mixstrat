from pathlib import Path

import pytest

from mixstrat.game.config import GameConfig, load_games
from mixstrat.game.tree import GameTree, Infoset
from mixstrat.llm.prompts import build_prompt, card_names, rules_text

PRESETS = Path(__file__).resolve().parents[2] / "configs" / "games.toml"
J, Q, K = 0, 1, 2


@pytest.fixture(scope="module")
def kuhn() -> GameTree:
    return GameTree(GameConfig(name="kuhn"))


def user_text(tree: GameTree, infoset: Infoset, **kwargs: object) -> str:
    return build_prompt(tree, infoset, **kwargs).messages[1]["content"]  # type: ignore[arg-type]


def test_card_names() -> None:
    assert card_names(GameConfig(name="k")) == ("J", "Q", "K")
    assert card_names(GameConfig.uniform("k13", 13))[-1] == "A"
    assert card_names(GameConfig.uniform("k5", 5)) == ("1", "2", "3", "4", "5")


def test_root_prompt(kuhn: GameTree) -> None:
    prompt = build_prompt(kuhn, Infoset(0, K, ()))
    text = prompt.messages[1]["content"]
    assert "Your card: K." in text
    assert "No one has acted yet." in text
    assert "A) check" in text
    assert "B) bet 1 chip (half the pot)" in text
    assert prompt.letters == ("A", "B")
    assert prompt.order == (0, 1)


def test_history_and_amounts(kuhn: GameTree) -> None:
    text = user_text(kuhn, Infoset(0, Q, ("check", "bet:0.5")))
    assert "- You checked." in text
    assert "- Your opponent bet 1 chip." in text
    assert "The pot is 3 chips: you have put in 1 chip and your opponent 2 chips." in text
    assert "A) fold" in text
    assert "B) call 1 chip" in text


def test_raise_amounts() -> None:
    tree = GameTree(GameConfig(name="r", max_raises=1, bet_sizes=(1.0,)))
    text = user_text(tree, Infoset(1, 0, ("bet:1",)))
    # Antes 1+1, player 0 bets 2. Player 1's raise: call 2 (pot 6), then add 6.
    assert "C) raise: put in 8 chips (2 to call plus 6 more)" in text


def test_rules_mention_variant_features() -> None:
    low = rules_text(GameConfig(name="l", payoff="low"), 0)
    assert "the lowest card wins" in low
    clair = GameConfig.uniform("c", 13, informed=(True, False))
    assert "Your opponent cannot see their own card" in rules_text(clair, 0)
    assert "You cannot see your own card" in rules_text(clair, 1)
    asym = GameConfig(name="a", deck=(1, 3), deck_p2=(1, 1))
    assert "Your deck has 4 cards: 1x 1, 3x 2" in rules_text(asym, 0)
    assert "Your deck has 2 cards: one card of each rank" in rules_text(asym, 1)


def test_uninformed_prompt_hides_card() -> None:
    tree = GameTree(GameConfig.uniform("c", 13, informed=(True, False)))
    text = user_text(tree, Infoset(1, None, ("check",)))
    assert "You cannot see your card." in text


def test_shuffle_is_a_deterministic_permutation() -> None:
    tree = GameTree(GameConfig.uniform("two", 13, bet_sizes=(0.5, 2.0)))
    orders = {build_prompt(tree, s, shuffle=True).order for s in tree.infosets_of(0)}
    assert len(orders) > 1  # different infosets get different orders
    for s in tree.infosets:
        prompt = build_prompt(tree, s, shuffle=True, seed=3)
        assert sorted(prompt.order) == list(range(len(tree.actions_at(s))))
        assert prompt.order == build_prompt(tree, s, shuffle=True, seed=3).order


def test_every_preset_infoset_has_a_prompt() -> None:
    for name, cfg in load_games(PRESETS).items():
        tree = GameTree(cfg)
        for s in tree.infosets:
            prompt = build_prompt(tree, s)
            text = prompt.messages[1]["content"]
            for letter in prompt.letters:
                assert f"\n{letter}) " in text, (name, str(s))
