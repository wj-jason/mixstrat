"""Chat prompts that show one information set to an LLM.

A prompt describes the game from the acting player's seat: the rules (derived
from the ``GameConfig``) in the system message, then the player's card, the
betting so far in chips, and the legal actions labelled A, B, C, ... so the
model can answer with a single token. Nothing here talks to a model.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from mixstrat.game.config import GameConfig
from mixstrat.game.tree import Action, GameTree, Infoset, Node

LETTERS = "ABCDEFGHIJ"
# Start of the assistant turn, so the next token is the answer letter (see ollama.py).
DEFAULT_PREFILL = "Answer: **"
Message = dict[str, str]

_FACES_3 = ("J", "Q", "K")
_FACES_13 = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")


@dataclass(frozen=True)
class Prompt:
    """Messages for one infoset, and how the offered letters map to tree actions."""

    infoset: Infoset
    messages: list[Message]
    letters: tuple[str, ...]  # letters offered, in display order
    order: tuple[int, ...]  # order[i] = index in node.actions of the action shown as letters[i]

    @property
    def option_lines(self) -> list[str]:
        """The lettered option lines exactly as shown to the model, e.g. "A) fold"."""
        starts = tuple(f"{letter}) " for letter in self.letters)
        return [
            line for line in self.messages[-1]["content"].splitlines() if line.startswith(starts)
        ]


def card_names(cfg: GameConfig) -> tuple[str, ...]:
    """Display names for ranks, lowest first."""
    n = cfg.num_ranks
    if n == 3:
        return _FACES_3
    if n == 13:
        return _FACES_13
    return tuple(str(r + 1) for r in range(n))


def chips(x: float) -> str:
    return f"{x:g} chip" if x == 1 else f"{x:g} chips"


def _pot_fraction(size: float) -> str:
    named = {0.25: "a quarter of the pot", 0.5: "half the pot", 1.0: "the size of the pot"}
    return named.get(size, f"{size:g}x the pot")


def _deck_text(deck: tuple[int, ...], names: tuple[str, ...]) -> str:
    total = sum(deck)
    if len(set(deck)) == 1:
        each = "one card" if deck[0] == 1 else f"{deck[0]} cards"
        return f"{total} cards: {each} of each rank"
    parts = ", ".join(f"{c}x {name}" for c, name in zip(deck, names, strict=True) if c)
    return f"{total} cards: {parts}"


def rules_text(cfg: GameConfig, player: int) -> str:
    """The rules of ``cfg`` as seen by ``player``."""
    names = card_names(cfg)
    opp = 1 - player
    winner = "lowest" if cfg.payoff == "low" else "highest"

    if cfg.deck_p2 is None:
        deal = (
            f"Each player is dealt one card from a single shuffled deck of "
            f"{_deck_text(cfg.deck, names)}. The two players get different cards from that deck."
        )
    else:
        decks = (cfg.deck, cfg.deck_p2)
        deal = (
            f"Each player is dealt one card from their own deck. Your deck has "
            f"{_deck_text(decks[player], names)}. Your opponent's deck has "
            f"{_deck_text(decks[opp], names)}."
        )

    sees = {
        (True, True): "You see your own card but not your opponent's.",
        (True, False): "You see your own card. Your opponent cannot see their own card or yours.",
        (
            False,
            True,
        ): "You cannot see your own card. Your opponent sees their own card but not yours.",
        (False, False): "Neither player can see any card until the showdown.",
    }[(cfg.informed[player], cfg.informed[opp])]

    order = "first" if cfg.first_actor == player else "second"
    bets = " or ".join(_pot_fraction(s) for s in cfg.bet_sizes)
    if cfg.max_raises == 0:
        facing = "Facing a bet, a player may fold or call. There is no raising."
    else:
        raises = " or ".join(_pot_fraction(s) for s in cfg.effective_raise_sizes)
        limit = "once" if cfg.max_raises == 1 else f"up to {cfg.max_raises} times"
        facing = (
            f"Facing a bet, a player may fold, call, or raise (match the bet, then add "
            f"{raises}). Raising is allowed {limit} per hand."
        )

    return "\n".join(
        [
            "You are playing a one-card poker game against one opponent.",
            "",
            f"Ranks from lowest to highest: {', '.join(names)}.",
            f"Deal: {deal}",
            f"Antes: before any betting you put in {chips(cfg.antes[player])} and your opponent "
            f"puts in {chips(cfg.antes[opp])}.",
            f"Information: {sees}",
            f"Betting: there is one betting round and you act {order}. A player who is not "
            f"facing a bet may check or bet {bets}. {facing} If both players check, or a bet "
            "is called, there is a showdown.",
            f"Showdown: the {winner} card wins the whole pot. Equal ranks split it.",
            "If a player folds, the other player wins the pot without a showdown.",
            "",
            "Your goal is to win as many chips as possible on average over many hands.",
        ]
    )


def path_to(tree: GameTree, node: Node) -> list[Node]:
    """Nodes from the root down to ``node``, inclusive."""
    path = [tree.root]
    for action in node.history:
        here = path[-1]
        path.append(tree.nodes[here.children[here.actions.index(action)]])
    return path


def _describe_action(action: Action, amount: float, to_call: float) -> str:
    """An action at a decision node, where the actor puts in ``amount`` more chips."""
    if action.kind == "bet":
        assert action.size is not None
        return f"bet {chips(amount)} ({_pot_fraction(action.size)})"
    if action.kind == "raise":
        return f"raise: put in {chips(amount)} ({to_call:g} to call plus {amount - to_call:g} more)"
    if action.kind == "call":
        return f"call {chips(amount)}"
    return action.kind


def _past_tense(action: Action, amount: float) -> str:
    return {
        "check": "checked",
        "bet": f"bet {chips(amount)}",
        "raise": f"raised, putting in {chips(amount)}",
        "call": f"called {chips(amount)}",
        "fold": "folded",
    }[action.kind]


def build_prompt(
    tree: GameTree,
    infoset: Infoset,
    *,
    shuffle: bool = False,
    seed: int = 0,
    reasoning: bool = False,
) -> Prompt:
    """The prompt for ``infoset``. ``shuffle`` permutes which letter each action gets,
    deterministically per (seed, infoset), to measure letter-position bias.
    ``reasoning`` asks the model to think briefly before answering."""
    cfg = tree.config
    node = tree.nodes[tree.infoset_node[infoset]]
    player = infoset.player
    names = card_names(cfg)

    lines: list[str] = []
    if infoset.card is None:
        lines.append("You cannot see your card.")
    else:
        lines.append(f"Your card: {names[infoset.card]}.")

    path = path_to(tree, node)
    if len(path) == 1:
        lines.append("No one has acted yet.")
    else:
        lines.append("Betting so far:")
        for here, nxt, action in zip(path, path[1:], node.history, strict=False):
            assert here.actor is not None
            who = "You" if here.actor == player else "Your opponent"
            put_in = nxt.contributions[here.actor] - here.contributions[here.actor]
            lines.append(f"- {who} {_past_tense(action, put_in)}.")

    mine, theirs = node.contributions[player], node.contributions[1 - player]
    pot = mine + theirs
    lines.append(
        f"The pot is {chips(pot)}: you have put in {chips(mine)} and your opponent {chips(theirs)}."
    )

    order = list(range(len(node.actions)))
    if shuffle:
        random.Random(f"{seed}:{infoset}").shuffle(order)
    letters = tuple(LETTERS[: len(order)])

    lines.append("")
    lines.append("Your options:")
    for letter, a in zip(letters, order, strict=True):
        child = tree.nodes[node.children[a]]
        amount = child.contributions[player] - mine
        lines.append(f"{letter}) {_describe_action(node.actions[a], amount, theirs - mine)}")
    lines.append("")
    if reasoning:
        lines.append("Think briefly: a few sentences at most, then answer.")
    lines.append(f"Reply with a single letter: {', '.join(letters)}.")

    messages = [
        {"role": "system", "content": rules_text(cfg, player)},
        {"role": "user", "content": "\n".join(lines)},
    ]
    return Prompt(infoset, messages, letters, tuple(order))
