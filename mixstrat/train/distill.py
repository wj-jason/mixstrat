"""Stage 1a of docs/post_training_design.md: distill the CFR strategy into the model.

Direct mode. At every infoset of the training game the model's answer
distribution (letter probabilities after the ``Answer: **`` prefill) is pushed
toward the CFR equilibrium's action distribution with a cross-entropy loss;
only LoRA adapter weights train. This is the pipeline check: training-game
exploitability should fall quickly. Held-out games are evaluated alongside to
show whether anything transfers; imitation is expected to transfer poorly.

    mixstrat-train --game kuhn_classic --held-out kuhn_lowball --steps 150 --track

Training prompts shuffle the answer letters every step (``--no-shuffle`` turns
this off) so the model learns actions, not letter positions; evaluation uses the
fixed order, like ``mixstrat-eval``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mixstrat.game.cfr import solve
from mixstrat.game.config import load_games
from mixstrat.game.strategy import Profile
from mixstrat.game.tree import GameTree
from mixstrat.llm.evaluate import Evaluation, evaluate, infoset_rows
from mixstrat.llm.hf import DEFAULT_MODEL, HFBackend
from mixstrat.llm.policy import PolicyResult, query_profile
from mixstrat.llm.prompts import build_prompt

PRESETS = Path(__file__).resolve().parents[2] / "configs" / "games.toml"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

log = logging.getLogger(__name__)


@dataclass
class Game:
    tree: GameTree
    equilibrium: Profile


def distill_step(
    backend: HFBackend,
    game: Game,
    optimizer: torch.optim.Optimizer,
    *,
    shuffle: bool,
    seed: int,
    micro_batch: int,
) -> tuple[float, float]:
    """One optimizer step on the mean cross-entropy from the equilibrium to the model.

    Infosets are processed ``micro_batch`` at a time with gradient accumulation,
    which gives the same gradient as one full batch in less memory. Returns the
    loss and the mean KL divergence (cross-entropy minus the equilibrium's
    entropy), which is 0 when the model matches exactly.
    """
    prompts = [build_prompt(game.tree, s, shuffle=shuffle, seed=seed) for s in game.tree.infosets]
    optimizer.zero_grad()
    total_ce, total_kl = 0.0, 0.0
    for start in range(0, len(prompts), micro_batch):
        chunk = prompts[start : start + micro_batch]
        outputs = backend.letter_logprobs([(p.messages, p.letters) for p in chunk])
        losses = []
        for prompt, out in zip(chunk, outputs, strict=True):
            target_np = game.equilibrium.at(prompt.infoset)[list(prompt.order)]
            target = torch.tensor(target_np, dtype=torch.float32, device=out.logprobs.device)
            ce = -(target * out.logprobs).sum()
            entropy = -float(sum(p * np.log(p) for p in target_np if p > 0))
            losses.append(ce)
            total_ce += ce.item()
            total_kl += ce.item() - entropy
        (torch.stack(losses).sum() / len(prompts)).backward()
    optimizer.step()
    return total_ce / len(prompts), total_kl / len(prompts)


def evaluate_game(backend: HFBackend, game: Game) -> tuple[PolicyResult, Evaluation]:
    result = query_profile(game.tree, backend)
    return result, evaluate(game.tree, result.profile, game.equilibrium)


def _eval_metrics(name: str, game: Game, result: PolicyResult, ev: Evaluation) -> dict[str, float]:
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Distill the CFR strategy into an LLM (LoRA).")
    parser.add_argument("--game", default="kuhn_classic", help="training game (preset name)")
    parser.add_argument(
        "--held-out", nargs="*", default=["kuhn_lowball"], help="games evaluated, not trained"
    )
    parser.add_argument("--games", type=Path, default=PRESETS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--micro-batch", type=int, default=4, help="infosets per forward pass")
    parser.add_argument(
        "--no-shuffle", action="store_true", help="fixed answer letters in training"
    )
    parser.add_argument("--cfr-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="adapter directory to save")
    parser.add_argument("--track", action="store_true", help="log the run to MLflow")
    parser.add_argument("--experiment", default="post-training")
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args(argv)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("mixstrat.train")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    from peft import LoraConfig, get_peft_model

    torch.manual_seed(args.seed)
    presets = load_games(args.games)
    names = [args.game, *[g for g in args.held_out if g != args.game]]
    for name in names:
        if name not in presets:
            parser.error(f"unknown game {name!r}; choose from {', '.join(presets)}")
    games = {}
    for name in names:
        tree = GameTree(presets[name])
        games[name] = Game(tree, solve(tree, iterations=args.cfr_iterations).profile)
    train = games[args.game]

    backend = HFBackend.load(args.model)
    lora = LoraConfig(
        r=args.lora_rank,
        lora_alpha=2 * args.lora_rank,
        lora_dropout=0.0,
        target_modules=LORA_TARGETS,
        task_type="CAUSAL_LM",
    )
    # Recompute activations in the backward pass instead of storing them: storing them
    # for 12 kuhn3 prompts needs ~8 GB, which overflows the GPU into system RAM.
    backend.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    backend.model.config.use_cache = False
    backend.model = get_peft_model(backend.model, lora)
    trainable = sum(p.numel() for p in backend.model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(
        [p for p in backend.model.parameters() if p.requires_grad], lr=args.lr
    )
    run_name = f"distill-{args.game}-{args.model.split('/')[-1]}"
    out = args.out or Path("outputs/checkpoints") / f"{run_name}-{time.strftime('%Y%m%d-%H%M%S')}"
    log.info(
        "training %s on %s (%d infosets), %.1fM LoRA params; held out: %s",
        args.model,
        args.game,
        len(train.tree.infosets),
        trainable / 1e6,
        ", ".join(names[1:]) or "none",
    )

    mlflow: Any = None  # the mlflow module when tracking
    if args.track:
        import os

        import mlflow as mlflow_module

        from mixstrat.llm.tracking import DEFAULT_TRACKING_URI

        mlflow = mlflow_module

        mlflow.set_tracking_uri(
            args.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
        )
        mlflow.set_experiment(args.experiment)
        mlflow.start_run(run_name=run_name)
        mlflow.set_tags({"stage": "1a-distill", "game": args.game})
        mlflow.log_params(
            {
                "model": args.model,
                "game": args.game,
                "held_out": ",".join(names[1:]),
                "objective": "distill-cfr",
                "steps": args.steps,
                "lr": args.lr,
                "lora_rank": args.lora_rank,
                "lora_targets": ",".join(LORA_TARGETS),
                "micro_batch": args.micro_batch,
                "trainable_params": trainable,
                "shuffle_train": not args.no_shuffle,
                "cfr_iterations": args.cfr_iterations,
                "seed": args.seed,
                "adapter_dir": str(out),
            }
        )
        log.info("MLflow run %s started", mlflow.active_run().info.run_id)

    def run_eval(step: int) -> dict[str, tuple[PolicyResult, Evaluation]]:
        backend.model.eval()
        results = {name: evaluate_game(backend, game) for name, game in games.items()}
        backend.model.train()
        metrics: dict[str, float] = {}
        for name, (result, ev) in results.items():
            metrics |= _eval_metrics(name, games[name], result, ev)
        if mlflow is not None:
            mlflow.log_metrics(metrics, step=step)
        log.info(
            "step %4d  eval  %s",
            step,
            "  ".join(f"{n} {100 * metrics[f'{n}/exploitability_norm']:.2f}%" for n in games),
        )
        return results

    try:
        start = time.perf_counter()
        results = run_eval(0)
        backend.model.train()
        for step in range(1, args.steps + 1):
            loss, kl = distill_step(
                backend,
                train,
                optimizer,
                shuffle=not args.no_shuffle,
                seed=step,
                micro_batch=args.micro_batch,
            )
            if mlflow is not None:
                mlflow.log_metrics({"train/loss": loss, "train/kl": kl}, step=step)
            if step % args.eval_every == 0 or step == args.steps:
                log.info("step %4d  loss %.4f  kl %.4f", step, loss, kl)
                results = run_eval(step)
        elapsed = time.perf_counter() - start

        backend.model.save_pretrained(out)
        log.info("saved adapter to %s (%.0fs)", out, elapsed)
        if mlflow is not None:
            from mixstrat.llm.tracking import RunLogger

            mlflow.log_metric("train_seconds", elapsed)
            run_id = mlflow.active_run().info.run_id
            for name, (result, ev) in results.items():
                game = games[name]
                rows = infoset_rows(game.tree, result, game.equilibrium)
                RunLogger(run_id, game.tree, f"{backend.name}+lora").finish(
                    result=result,
                    equilibrium=game.equilibrium,
                    evaluation=ev,
                    rows=rows,
                    metrics={},
                    prefix=f"final/{name}/",
                )
            mlflow.end_run()
            log.info("logged MLflow run %s", run_id)
    except BaseException:
        if mlflow is not None:
            mlflow.end_run(status="FAILED")
        raise


if __name__ == "__main__":
    main()
