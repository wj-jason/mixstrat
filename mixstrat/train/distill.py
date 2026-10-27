"""Stage 1a of docs/post_training_design.md: distill the CFR strategy into the model.

Direct mode. At every infoset of the training game the model's answer
distribution (letter probabilities after the ``Answer: **`` prefill) is pushed
toward the CFR equilibrium's action distribution with a cross-entropy loss;
only LoRA adapter weights train. The loss also penalizes probability on tokens
other than the offered letters; without that term the model can match the
letter ratios while leaking mass to invalid answers. This is the pipeline
check: training-game exploitability should fall quickly, while imitation is
expected to transfer poorly to held-out games.

    mixstrat-train --objective distill --game kuhn_classic --held-out kuhn_lowball --track
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from mixstrat.llm.hf import HFBackend
from mixstrat.llm.prompts import build_prompt
from mixstrat.train.common import Item


def distill_step(
    backend: HFBackend,
    items: Sequence[Item],
    optimizer: torch.optim.Optimizer,
    *,
    shuffle: bool,
    seed: int,
    micro_batch: int,
    mass_weight: float = 1.0,
) -> dict[str, float]:
    """One optimizer step on ``items`` (infosets of one or more games). Per infoset the loss is

        cross-entropy(equilibrium, renormalized letters) + mass_weight * -log(legal mass)

    With ``mass_weight=1`` this is the cross-entropy against the model's full
    next-token distribution: probability on any other token counts as wrong.
    Infosets are processed ``micro_batch`` at a time with gradient accumulation,
    which gives the same gradient as one full batch in less memory.

    Returns ``loss``, ``kl`` (from the equilibrium to the renormalized letter
    distribution), ``leak`` (mean -log legal mass) and ``legal_mass_min``.
    """
    prompts = [build_prompt(game.tree, s, shuffle=shuffle, seed=seed) for game, s in items]
    targets = [game.equilibrium.at(s) for game, s in items]
    optimizer.zero_grad()
    total_loss, total_kl, total_leak, min_mass = 0.0, 0.0, 0.0, 1.0
    for start in range(0, len(prompts), micro_batch):
        chunk = prompts[start : start + micro_batch]
        outputs = backend.letter_logprobs([(p.messages, p.letters) for p in chunk])
        losses = []
        for prompt, out, eq in zip(
            chunk, outputs, targets[start : start + micro_batch], strict=True
        ):
            target_np = eq[list(prompt.order)]
            target = torch.tensor(target_np, dtype=torch.float32, device=out.logprobs.device)
            ce = -(target * out.logprobs).sum()
            leak = -out.log_legal_mass
            loss = ce + mass_weight * leak
            entropy = -float(sum(p * np.log(p) for p in target_np if p > 0))
            losses.append(loss)
            total_loss += loss.item()
            total_kl += ce.item() - entropy
            total_leak += leak.item()
            min_mass = min(min_mass, out.legal_mass.item())
        (torch.stack(losses).sum() / len(prompts)).backward()
    optimizer.step()
    n = len(prompts)
    return {
        "loss": total_loss / n,
        "kl": total_kl / n,
        "leak": total_leak / n,
        "legal_mass_min": min_mass,
    }
