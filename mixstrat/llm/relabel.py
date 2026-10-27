"""Relabelled views of a game: the same strategic game under other card names.

Two changes leave a game strategically identical:

- renaming the ranks (``card_names``);
- a *flip*: list the ranks in reverse and reverse the payoff rule ("highest
  wins" becomes "lowest wins"). Original rank ``i`` becomes listed rank
  ``n - 1 - i``, keeps its name, and stays exactly as strong.

Training on random views (``Relabeler``) means the right answer can only come
from the rank list and showdown rule, not from a card's name. The probe label
sets (``PROBE_NAMES``) never appear in random views, so they stay held out.
"""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np

from mixstrat.game.config import GameConfig
from mixstrat.game.tree import GameTree, Infoset

NEUTRAL_PROBE = ("Vex", "Dorn", "Pim")
PROBE_NAMES = frozenset({"J", "Q", "K", "2", "3", "4", *NEUTRAL_PROBE})


def _cvc_words() -> tuple[str, ...]:
    consonants, vowels = "bdfgklmnprstvz", "aeiou"
    words = (c1.upper() + v + c2 for c1 in consonants for v in vowels for c2 in consonants)
    return tuple(w for w in words if w not in PROBE_NAMES)


# Each pool has at least 13 names, enough for a 13-rank game. Numbers are drawn in
# random order, so their numeric order usually disagrees with the rank list.
NAME_POOLS: dict[str, tuple[str, ...]] = {
    "words": _cvc_words(),
    "numbers": tuple(str(n) for n in range(5, 100)),
    "greek": (
        "Alpha",
        "Beta",
        "Gamma",
        "Delta",
        "Epsilon",
        "Zeta",
        "Eta",
        "Theta",
        "Iota",
        "Kappa",
        "Lambda",
        "Mu",
        "Nu",
        "Xi",
        "Omicron",
        "Pi",
        "Rho",
        "Sigma",
        "Tau",
        "Upsilon",
        "Phi",
        "Chi",
        "Psi",
        "Omega",
    ),
    "colors": (
        "Red",
        "Blue",
        "Green",
        "Yellow",
        "Purple",
        "Orange",
        "Pink",
        "Brown",
        "Black",
        "White",
        "Gray",
        "Teal",
        "Gold",
        "Silver",
        "Violet",
        "Cyan",
    ),
    "animals": (
        "Fox",
        "Owl",
        "Elk",
        "Bat",
        "Cat",
        "Dog",
        "Yak",
        "Eel",
        "Ant",
        "Bee",
        "Cow",
        "Pig",
        "Rat",
        "Hen",
        "Ape",
        "Emu",
    ),
}
assert not any(PROBE_NAMES & set(pool) for pool in NAME_POOLS.values())


def relabel_config(cfg: GameConfig, names: tuple[str, ...], flip: bool) -> GameConfig:
    """``cfg`` with ``names[i]`` naming original rank ``i``, optionally flipped."""
    if len(names) != cfg.num_ranks:
        raise ValueError(f"need {cfg.num_ranks} names, got {len(names)}")
    if not flip:
        return replace(cfg, card_names=tuple(names))
    return replace(
        cfg,
        card_names=tuple(reversed(names)),
        deck=tuple(reversed(cfg.deck)),
        deck_p2=None if cfg.deck_p2 is None else tuple(reversed(cfg.deck_p2)),
        payoff="low" if cfg.payoff == "high" else "high",
    )


def map_infoset(infoset: Infoset, num_ranks: int, flip: bool) -> Infoset:
    """Where ``infoset`` of the original game sits in the (possibly flipped) view."""
    if not flip or infoset.card is None:
        return infoset
    return Infoset(infoset.player, num_ranks - 1 - infoset.card, infoset.history)


def relabeled_tree(tree: GameTree, names: tuple[str, ...], flip: bool) -> GameTree:
    """A copy of ``tree`` whose prompts show the relabelled rules. For prompting only:
    the betting tree is shared, but payoffs and deal probabilities are the original
    game's (prompts read neither)."""
    view = copy.copy(tree)
    view.config = relabel_config(tree.config, names, flip)
    return view


def random_names(n: int, rng: np.random.Generator) -> tuple[str, ...]:
    """``n`` distinct names from one random pool, in random order."""
    pool = NAME_POOLS[str(rng.choice(sorted(NAME_POOLS)))]
    return tuple(str(pool[i]) for i in rng.choice(len(pool), size=n, replace=False))


class Relabeler:
    """Shows each training infoset under a random view with probability ``prob``
    (flipped with probability ``flip_prob``); otherwise under the game's own labels."""

    def __init__(self, prob: float, flip_prob: float, rng: np.random.Generator) -> None:
        self.prob, self.flip_prob, self.rng = prob, flip_prob, rng

    def __call__(self, tree: GameTree, infoset: Infoset) -> tuple[GameTree, Infoset]:
        if self.rng.random() >= self.prob:
            return tree, infoset
        n = tree.config.num_ranks
        flip = bool(self.rng.random() < self.flip_prob)
        view = relabeled_tree(tree, random_names(n, self.rng), flip)
        return view, map_infoset(infoset, n, flip)
