"""Stage 1b of docs/post_training_design.md: train on exact chip values.

Direct mode. At each infoset the model's answer distribution ``p`` is scored by
the counterfactual action values ``q(I, a)`` (``mixstrat.game.values``) under a
profile in which the model plays its own seat, including its later decisions,
and the opponent plays either

- the CFR equilibrium (``opponent="cfr"``, stage 1b-i), or
- the model itself (``opponent="self"``, self-play, stage 1b-ii).

Per infoset the loss is

    policy term  +  beta * KL(p || magnet)  +  mass_weight * -log(legal mass)

averaged over the infosets the opponent's play can reach, with the policy term
chosen by ``update``:

- ``"neurd"`` (default): ``-sum_a sg(q_a - mean(q)) log p_a``, Neural
  Replicator Dynamics (Hennes et al. 2020). It moves each action's logit by its
  advantage, without the ``p_a`` factor of the softmax policy gradient, so
  actions the model currently almost never takes can still recover, and every
  infoset pushes in proportion to how much better its actions are. It stops
  where the actions in play are worth the same: the equilibrium condition.
  Unchecked, it also keeps pushing losing actions further below zero
  probability forever (log p -> -inf), and those pushes crowd out the
  infosets that actually need to change; like the paper's logit threshold, an
  action is no longer pushed down once ``p < floor`` or up once ``p > 1 - floor``.
- ``"expected"``: ``-sum_a p_a q_a``, the plain expected value. Its logit
  gradient ``p_a (q_a - sum_b p_b q_b)`` vanishes as ``p_a -> 0``; in practice a
  near-deterministic model stops learning, and infosets with large gradients
  drag the shared LoRA weights for all the others.

Either way the per-infoset direction is the local improvement CFR follows, not
the gradient of total EV, so infosets the model's own play rarely reaches
keep learning.

The KL term is the magnet of magnetic mirror descent (Sokota et al. 2022),
which makes policy-gradient self-play converge in two-player zero-sum games
instead of cycling. ``magnet="uniform"`` (the default) is an entropy bonus and
converges to a regularized equilibrium that approaches Nash as ``beta`` shrinks.
``magnet="reference"`` pulls toward the starting model instead (LoRA disabled);
for qwen3 that is the "bet and call everything" policy, so it biases training
toward it.

In self-play the values can be averaged over the last ``window`` steps:
counterfactual values are linear in the opponent's reach probabilities, so
averaging ``q * reach`` and ``reach`` separately gives the values against the
average of the recent opponents (the model's own later decisions are averaged
too, approximately).

    mixstrat-train --objective value-self-play --game kuhn_classic --track
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch

from mixstrat.game.strategy import Profile
from mixstrat.game.tree import Infoset
from mixstrat.game.values import ActionValues, counterfactual_values
from mixstrat.llm.hf import HFBackend
from mixstrat.llm.prompts import build_prompt
from mixstrat.train.common import Game, Item, model_profile

Opponent = Literal["cfr", "self"]
Magnet = Literal["uniform", "reference"]
Update = Literal["neurd", "expected"]


def training_values(game: Game, model: Profile, opponent: Opponent) -> dict[Infoset, ActionValues]:
    """Action values for both seats: the model in its own seat, ``opponent`` in the other."""
    values: dict[Infoset, ActionValues] = {}
    for player in (0, 1):
        if opponent == "self":
            profile = model
        elif player == 0:
            profile = Profile.combine(model, game.equilibrium)
        else:
            profile = Profile.combine(game.equilibrium, model)
        values |= counterfactual_values(game.tree, profile, player)
    return values


def average_values(history: deque[dict[Infoset, ActionValues]]) -> dict[Infoset, ActionValues]:
    averaged = {}
    for infoset in history[-1]:
        reaches = np.array([h[infoset].reach for h in history])
        total = float(reaches.sum())
        if total > 0:
            q = sum(h[infoset].q * r for h, r in zip(history, reaches, strict=True)) / total
        else:
            q = history[-1][infoset].q
        averaged[infoset] = ActionValues(np.asarray(q), total / len(history))
    return averaged


def value_step(
    backend: HFBackend,
    items: Sequence[Item],
    optimizer: torch.optim.Optimizer,
    *,
    opponent: Opponent,
    shuffle: bool,
    seed: int,
    micro_batch: int,
    beta: float = 0.02,
    magnet: Magnet = "uniform",
    update: Update = "neurd",
    floor: float = 1e-3,
    mass_weight: float = 1.0,
    history: dict[str, deque[dict[Infoset, ActionValues]]] | None = None,
    cache: dict[str, dict[Infoset, ActionValues]] | None = None,
    refresh: bool = True,
    profile_batch: int = 16,
) -> dict[str, float]:
    """One optimizer step on the value objective (see the module docstring), on
    ``items`` (infosets of one or more games).

    ``history`` maps game names to the recent values to average over (self-play
    with a window); pass None to use only the current values. With a ``cache``
    and ``refresh=False`` the values from the last refresh are reused instead of
    re-querying the model (``cache`` is updated on every refresh). The strategy
    snapshot runs without gradients in batches of ``profile_batch``. Returns ``loss``,
    ``value`` (mean expected chips of the model's answers under the training
    values), ``kl_magnet``, ``leak`` and ``legal_mass_min``.
    """
    # The model's current strategy in each game (fixed answer letters) defines the values.
    games: dict[str, Game] = {game.name: game for game, _ in items}
    values: dict[str, dict[Infoset, ActionValues]] = {}
    for name, game in games.items():
        if not refresh and cache is not None and name in cache:
            values[name] = cache[name]
            continue
        model = model_profile(backend, game.tree, profile_batch)
        values[name] = training_values(game, model, opponent)
        if history is not None:
            history[name].append(values[name])
            values[name] = average_values(history[name])
        if cache is not None:
            cache[name] = values[name]

    item_values = [values[game.name][s] for game, s in items]
    active = [
        (build_prompt(game.tree, s, shuffle=shuffle, seed=seed), v)
        for (game, s), v in zip(items, item_values, strict=True)
        if v.reach > 0
    ]
    batch = [(p.messages, p.letters) for p, _ in active]

    reference: list[torch.Tensor] = []
    if magnet == "reference" and beta > 0:
        with torch.no_grad(), backend.model.disable_adapter():
            for start in range(0, len(batch), micro_batch):
                outs = backend.letter_logprobs(batch[start : start + micro_batch])
                reference += [out.logprobs for out in outs]

    optimizer.zero_grad()
    totals = {"loss": 0.0, "value": 0.0, "kl_magnet": 0.0, "leak": 0.0}
    min_mass = 1.0
    for start in range(0, len(active), micro_batch):
        chunk = active[start : start + micro_batch]
        outputs = backend.letter_logprobs(batch[start : start + micro_batch])
        losses = []
        for i, ((prompt, v), out) in enumerate(zip(chunk, outputs, strict=True)):
            device = out.logprobs.device
            q = torch.tensor(v.q[list(prompt.order)], dtype=torch.float32, device=device)
            p = out.logprobs.exp()
            value = (p * q).sum()
            advantage = q - q.mean()
            saturated = ((advantage < 0) & (p < floor)) | ((advantage > 0) & (p > 1 - floor))
            neurd = -(torch.where(saturated, 0.0, advantage) * out.logprobs).sum()
            policy_term = neurd if update == "neurd" else -value
            if magnet == "reference" and beta > 0:
                magnet_logprobs = reference[start + i]
            else:
                magnet_logprobs = torch.full_like(out.logprobs, -math.log(len(prompt.letters)))
            kl = (p * (out.logprobs - magnet_logprobs)).sum()
            leak = -out.log_legal_mass
            loss = policy_term + beta * kl + mass_weight * leak
            losses.append(loss)
            for key, x in (("loss", loss), ("value", value), ("kl_magnet", kl), ("leak", leak)):
                totals[key] += x.item()
            min_mass = min(min_mass, out.legal_mass.item())
        (torch.stack(losses).sum() / len(active)).backward()
    optimizer.step()
    return {k: v / len(active) for k, v in totals.items()} | {"legal_mass_min": min_mass}
