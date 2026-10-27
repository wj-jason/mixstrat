from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from mixstrat.game.cfr import solve
from mixstrat.game.config import GameConfig
from mixstrat.game.strategy import Profile
from mixstrat.game.tree import GameTree, Infoset
from mixstrat.llm.backend import (
    BackendError,
    ChoiceDistribution,
    ReasoningSample,
    letter_distribution,
)
from mixstrat.llm.evaluate import evaluate, infoset_rows
from mixstrat.llm.policy import ChoiceCache, query_profile
from mixstrat.llm.prompts import Message, build_prompt
from mixstrat.llm.report import card_ranges, format_ranges, render_reasoning, render_report

# ---------------------------------------------------------------- letter distributions


def test_letter_distribution_strips_and_renormalizes() -> None:
    top = [
        {"token": "B", "logprob": np.log(0.6)},
        {"token": " A", "logprob": np.log(0.2)},
        {"token": "A", "logprob": np.log(0.1)},
        {"token": "Sure", "logprob": np.log(0.1)},
    ]
    dist = letter_distribution(top, ["A", "B", "C"])
    assert dist.legal_mass == pytest.approx(0.9)
    assert np.allclose(dist.probs, [1 / 3, 2 / 3, 0.0])


def test_letter_distribution_needs_a_letter() -> None:
    with pytest.raises(BackendError):
        letter_distribution([{"token": "Hmm", "logprob": 0.0}], ["A", "B"])


# ---------------------------------------------------------------- query_profile


class ProfileBackend:
    """Answers each prompt with a fixed profile's probabilities, in letter order."""

    def __init__(self, tree: GameTree, profile: Profile, shuffle: bool) -> None:
        self.calls = 0
        self._answers = {}
        for s in tree.infosets:
            prompt = build_prompt(tree, s, shuffle=shuffle)
            probs = profile.at(s)[list(prompt.order)]
            self._answers[ChoiceCache.key("", prompt.messages, prompt.letters)] = probs

    @property
    def name(self) -> str:
        return "fake/profile"

    def choose(self, messages: Sequence[Message], letters: Sequence[str]) -> ChoiceDistribution:
        self.calls += 1
        return ChoiceDistribution(self._answers[ChoiceCache.key("", messages, letters)], 1.0)


@pytest.fixture(scope="module")
def tree() -> GameTree:
    return GameTree(GameConfig.uniform("k5", 5, bet_sizes=(0.5, 1.0)))


@pytest.fixture(scope="module")
def equilibrium(tree: GameTree) -> Profile:
    return solve(tree, iterations=1000).profile


@pytest.mark.parametrize("shuffle", [False, True])
def test_query_recovers_backend_strategy(
    tree: GameTree, equilibrium: Profile, shuffle: bool
) -> None:
    backend = ProfileBackend(tree, equilibrium, shuffle)
    result = query_profile(tree, backend, shuffle=shuffle)
    assert backend.calls == len(tree.infosets)
    for nid in tree.decision_ids:
        assert np.allclose(result.profile.probs[nid], equilibrium.probs[nid])


def test_cache_skips_backend(tree: GameTree, equilibrium: Profile, tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    backend = ProfileBackend(tree, equilibrium, shuffle=False)
    query_profile(tree, backend, cache=ChoiceCache(path))
    backend.calls = 0
    again = query_profile(tree, backend, cache=ChoiceCache(path))  # reloaded from disk
    assert backend.calls == 0
    assert again.num_cached == len(tree.infosets)


def test_evaluate_equilibrium_against_itself(tree: GameTree, equilibrium: Profile) -> None:
    backend = ProfileBackend(tree, equilibrium, shuffle=False)
    result = query_profile(tree, backend)
    ev = evaluate(tree, result.profile, equilibrium)
    assert ev.seat_loss == pytest.approx((0.0, 0.0), abs=1e-9)
    assert ev.llm.exploitability == pytest.approx(ev.equilibrium.exploitability)
    assert ev.uniform.exploitability > ev.llm.exploitability
    rows = infoset_rows(tree, result, equilibrium)
    assert len(rows) == len(tree.infosets)
    assert max(r["tv"] for r in rows) == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------- report


def test_card_ranges_on_kuhn_equilibrium() -> None:
    tree = GameTree(GameConfig(name="kuhn"))
    eq = solve(tree, iterations=3000).profile
    facing_bet = tree.infoset_node[Infoset(1, 0, ("bet:0.5",))]
    # Player 1 facing a bet: fold J, call Q one third of the time, call K.
    ranges = card_ranges(tree, eq, facing_bet)
    assert [(r.start, r.end, r.mixed) for r in ranges] == [
        ("J", "J", False),
        ("Q", "Q", True),
        ("K", "K", False),
    ]
    assert ranges[1].probs[1] == pytest.approx(1 / 3, abs=0.03)
    assert str(ranges[0]) == "J fold"
    assert str(ranges[1]).startswith("Q fold 0.6")


def test_card_ranges_merge_similar_cards() -> None:
    tree = GameTree(GameConfig.uniform("k5", 5))
    root = tree.root.id
    probs = Profile.uniform(tree).probs
    probs[root] = np.array([[0.6, 0.4], [0.5, 0.5], [1.0, 0.0], [1.0, 0.0], [0.4, 0.6]])
    ranges = card_ranges(tree, Profile(tree, probs), root)
    assert (
        format_ranges(ranges)
        == "1-2 check 0.55 / bet:0.5 0.45 | 3-4 check | 5 check 0.40 / bet:0.5 0.60"
    )


def test_render_report(tree: GameTree, equilibrium: Profile) -> None:
    result = query_profile(tree, ProfileBackend(tree, equilibrium, shuffle=False))
    ev = evaluate(tree, result.profile, equilibrium)
    page = render_report(
        tree, result, equilibrium, ev, infoset_rows(tree, result, equilibrium), "fake"
    )
    assert page.startswith("<!doctype html>")
    assert page.count('<section class="node">') == len(tree.decision_ids)


# ---------------------------------------------------------------- reasoning


class ReasoningProfileBackend(ProfileBackend):
    """Like ProfileBackend, but answers via two reasoning samples that average to the profile."""

    def choose(self, messages: Sequence[Message], letters: Sequence[str]) -> ChoiceDistribution:
        target = super().choose(messages, letters).probs
        first = np.eye(len(letters))[0]
        second = 2 * target - first  # mean of (first, second) is the target
        if second.min() < 0:
            first = second = target
        samples = tuple(
            ReasoningSample(f"trace {i} for {letters}", p, 1.0, 100 * (i + 1), truncated=i == 1)
            for i, p in enumerate((first, second))
        )
        return ChoiceDistribution(np.mean([first, second], axis=0), 1.0, samples)


def test_reasoning_prompt_asks_for_brief_thinking(tree: GameTree) -> None:
    s = tree.infosets[0]
    assert "Think briefly" in build_prompt(tree, s, reasoning=True).messages[1]["content"]
    assert "Think briefly" not in build_prompt(tree, s).messages[1]["content"]


def test_reasoning_samples_survive_shuffle_and_cache(
    tree: GameTree, equilibrium: Profile, tmp_path: Path
) -> None:
    backend = ReasoningProfileBackend(tree, equilibrium, shuffle=True)
    path = tmp_path / "cache.jsonl"
    first = query_profile(tree, backend, shuffle=True, cache=ChoiceCache(path))
    again = query_profile(tree, backend, shuffle=True, cache=ChoiceCache(path))
    assert again.num_cached == len(tree.infosets)
    for result in (first, again):
        assert result.truncated_rate == pytest.approx(0.5)
        assert result.mean_think_tokens == pytest.approx(150)
        for q in result.queries:
            assert len(q.samples) == 2
            mean = np.mean([q.action_probs(s.probs) for s in q.samples], axis=0)
            assert np.allclose(mean, equilibrium.at(q.infoset))  # samples map to tree actions
    assert again.queries[3].samples[0].thinking == first.queries[3].samples[0].thinking


def test_render_reasoning(tree: GameTree, equilibrium: Profile) -> None:
    result = query_profile(tree, ReasoningProfileBackend(tree, equilibrium, shuffle=False))
    page = render_reasoning(tree, result, equilibrium, "fake")
    assert page.count('<section class="infoset">') == len(tree.infosets)
    assert page.count("<details>") == 2 * len(tree.infosets)
    assert "cut off" in page
    assert "A) check" in page and "B) bet 1 chip" in page  # options as the model saw them
