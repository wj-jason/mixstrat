"""Pieces shared by the training objectives: games, sampling, evaluation and settings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mixstrat.game.strategy import Profile
from mixstrat.game.tree import FloatArray, GameTree, Infoset
from mixstrat.llm.evaluate import Evaluation, evaluate
from mixstrat.llm.hf import HFBackend
from mixstrat.llm.policy import PolicyResult, query_profile, to_action_order
from mixstrat.llm.prompts import build_prompt

PRESETS = Path(__file__).resolve().parents[2] / "configs" / "games.toml"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
MIN_LEGAL_MASS = 0.95  # below this at any infoset, evaluation logs a warning


@dataclass
class Game:
    tree: GameTree
    equilibrium: Profile
    name: str = ""


Item = tuple[Game, Infoset]  # one training example: an infoset of a game


def sample_items(games: Sequence[Game], batch: int, rng: np.random.Generator) -> list[Item]:
    """Infosets to train on this step: all of them if ``batch <= 0``, otherwise
    ``batch`` split evenly across the games (so a 12-infoset game isn't drowned out
    by a 52-infoset one), each game's share drawn without replacement."""
    if batch <= 0:
        return [(game, s) for game in games for s in game.tree.infosets]
    share = max(1, batch // len(games))
    items: list[Item] = []
    for game in games:
        infosets = game.tree.infosets
        picks = rng.choice(len(infosets), size=min(share, len(infosets)), replace=False)
        items += [(game, infosets[i]) for i in sorted(picks)]
    return items


def model_profile(
    backend: HFBackend, tree: GameTree, micro_batch: int, *, shuffle: bool = False, seed: int = 0
) -> Profile:
    """The model's current strategy (fixed answer letters unless ``shuffle``), batched and
    without gradients."""
    prompts = [build_prompt(tree, s, shuffle=shuffle, seed=seed) for s in tree.infosets]
    table: dict[Infoset, FloatArray] = {}
    with torch.no_grad():
        for start in range(0, len(prompts), micro_batch):
            chunk = prompts[start : start + micro_batch]
            outputs = backend.letter_logprobs([(p.messages, p.letters) for p in chunk])
            for prompt, out in zip(chunk, outputs, strict=True):
                probs = out.logprobs.exp().cpu().numpy().astype(np.float64)
                table[prompt.infoset] = to_action_order(prompt, probs / probs.sum())
    return Profile.from_infosets(tree, table)


def evaluate_game(backend: HFBackend, game: Game) -> tuple[PolicyResult, Evaluation]:
    result = query_profile(game.tree, backend)
    return result, evaluate(game.tree, result.profile, game.equilibrium)


def eval_metrics(name: str, game: Game, result: PolicyResult, ev: Evaluation) -> dict[str, float]:
    tv = [
        0.5 * float(np.abs(q.probs - game.equilibrium.at(q.infoset)).sum()) for q in result.queries
    ]
    return {
        f"{name}/exploitability_norm": ev.llm.normalized(game.tree.config.initial_pot),
        f"{name}/seat0_ev_vs_eq": ev.seat_ev[0],
        f"{name}/seat1_ev_vs_eq": ev.seat_ev[1],
        f"{name}/tv_mean": float(np.mean(tv)),
        f"{name}/legal_mass_min": result.min_legal_mass,
    }
