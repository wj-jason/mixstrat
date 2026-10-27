"""Command line for post-training (see docs/post_training_design.md).

    mixstrat-train --objective distill          --game kuhn_classic --track   # stage 1a
    mixstrat-train --objective value-vs-cfr     --game kuhn_classic --track   # stage 1b-i
    mixstrat-train --objective value-self-play  --game kuhn_classic --track   # stage 1b-ii
    mixstrat-train --game kuhn_classic kuhn13 kuhn13_lowball --batch-infosets 24 --track

Several ``--game`` values train on all of them; ``--batch-infosets`` then samples
that many infosets per step, split evenly across the games.

Direct mode, qwen3 via Hugging Face with LoRA. Training prompts shuffle the
answer letters every step (``--no-shuffle`` turns this off) so the model learns
actions, not letter positions; evaluation uses the fixed order, like
``mixstrat-eval``. Held-out games are evaluated alongside but never trained on.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mixstrat.game.cfr import solve
from mixstrat.game.config import load_games
from mixstrat.game.tree import GameTree, Infoset
from mixstrat.game.values import ActionValues
from mixstrat.llm.evaluate import Evaluation, infoset_rows
from mixstrat.llm.hf import DEFAULT_MODEL, HFBackend
from mixstrat.llm.policy import PolicyResult
from mixstrat.llm.relabel import Relabeler
from mixstrat.train.common import (
    LORA_TARGETS,
    MIN_LEGAL_MASS,
    PRESETS,
    Game,
    eval_metrics,
    evaluate_game,
    sample_items,
)
from mixstrat.train.distill import distill_step
from mixstrat.train.value import value_step

OBJECTIVES = {
    "distill": "1a-distill",
    "value-vs-cfr": "1b-value-vs-cfr",
    "value-self-play": "1b-value-self-play",
}

# Named explicitly: under `python -m` (how mixstrat-sweep runs it) __name__ is "__main__",
# which the "mixstrat.train" handler set up in main() would not see.
log = logging.getLogger("mixstrat.train.cli")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Post-train an LLM on a poker variant (LoRA).")
    parser.add_argument("--objective", choices=list(OBJECTIVES), default="distill")
    parser.add_argument(
        "--game", nargs="+", default=["kuhn_classic"], help="training game(s) (preset names)"
    )
    parser.add_argument(
        "--batch-infosets",
        type=int,
        default=0,
        help="infosets per step, split evenly across training games (0 = all)",
    )
    parser.add_argument(
        "--held-out", nargs="*", default=["kuhn_lowball"], help="games evaluated, not trained"
    )
    parser.add_argument("--games", type=Path, default=PRESETS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="default 1e-4 (distill) or 3e-5 (value objectives: 1e-4 saturates in one Adam step)",
    )
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--micro-batch", type=int, default=4, help="infosets per forward pass")
    parser.add_argument(
        "--mass-weight",
        type=float,
        default=1.0,
        help="weight on -log(legal mass); 1 = cross-entropy over the full vocabulary",
    )
    parser.add_argument("--beta", type=float, default=0.02, help="value objectives: KL weight")
    parser.add_argument(
        "--magnet",
        choices=("uniform", "reference"),
        default="uniform",
        help="value objectives: policy the KL term pulls toward",
    )
    parser.add_argument(
        "--window", type=int, default=1, help="value-self-play: average values over N steps"
    )
    parser.add_argument(
        "--value-refresh",
        type=int,
        default=1,
        help="value objectives: recompute the model's strategy and values every N steps",
    )
    parser.add_argument(
        "--update",
        choices=("neurd", "expected"),
        default="neurd",
        help="value objectives: policy update (see mixstrat/train/value.py)",
    )
    parser.add_argument(
        "--neurd-floor",
        type=float,
        default=1e-3,
        help="neurd: stop pushing an action below this probability (or above 1 minus it)",
    )
    parser.add_argument(
        "--no-shuffle", action="store_true", help="fixed answer letters in training"
    )
    parser.add_argument(
        "--relabel",
        type=float,
        default=0.0,
        help="distill: show each infoset with random card names this often (see llm/relabel.py)",
    )
    parser.add_argument(
        "--relabel-flip",
        type=float,
        default=0.5,
        help="distill: of the relabelled prompts, this share also list ranks in reverse "
        "and flip the payoff rule",
    )
    parser.add_argument("--cfr-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="adapter directory to save")
    parser.add_argument("--track", action="store_true", help="log the run to MLflow")
    parser.add_argument("--experiment", default="post-training")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--run-name", default=None, help="MLflow run name (default: generated)")
    parser.add_argument(
        "--tag", action="append", default=[], metavar="KEY=VALUE", help="extra MLflow tag"
    )
    args = parser.parse_args(argv)
    tags = {}
    for tag in args.tag:
        key, sep, value = tag.partition("=")
        if not sep:
            parser.error(f"--tag needs KEY=VALUE, got {tag!r}")
        tags[key] = value
    if args.lr is None:
        args.lr = 1e-4 if args.objective == "distill" else 3e-5
    if args.relabel and args.objective != "distill":
        parser.error("--relabel is only supported with --objective distill")

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("mixstrat.train")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    from peft import LoraConfig, get_peft_model

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    presets = load_games(args.games)
    held_out = [g for g in args.held_out if g not in args.game]
    names = [*args.game, *held_out]
    for name in names:
        if name not in presets:
            parser.error(f"unknown game {name!r}; choose from {', '.join(presets)}")
    games = {}
    for name in names:
        tree = GameTree(presets[name])
        games[name] = Game(tree, solve(tree, iterations=args.cfr_iterations).profile, name)
    train = [games[name] for name in args.game]
    n_train_infosets = sum(len(g.tree.infosets) for g in train)

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
    value_objective = args.objective != "distill"
    value_cache: dict[str, dict[Infoset, ActionValues]] = {}
    history: dict[str, deque[dict[Infoset, ActionValues]]] | None = None
    if args.objective == "value-self-play":
        history = {g.name: deque(maxlen=args.window) for g in train}

    run_name = (
        args.run_name or f"{args.objective}-{'+'.join(args.game)}-{args.model.split('/')[-1]}"
    )
    out = args.out or Path("outputs/checkpoints") / f"{run_name}-{time.strftime('%Y%m%d-%H%M%S')}"
    log.info(
        "%s: training %s on %s (%d infosets, %s per step), %.1fM LoRA params; held out: %s",
        args.objective,
        args.model,
        ", ".join(args.game),
        n_train_infosets,
        args.batch_infosets if args.batch_infosets > 0 else "all",
        trainable / 1e6,
        ", ".join(held_out) or "none",
    )

    mlflow: Any = None  # the mlflow module when tracking
    if args.track:
        import mlflow as mlflow_module

        from mixstrat.llm.tracking import DEFAULT_TRACKING_URI

        mlflow = mlflow_module
        mlflow.set_tracking_uri(
            args.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
        )
        mlflow.set_experiment(args.experiment)
        mlflow.start_run(run_name=run_name)
        mlflow.set_tags({"stage": OBJECTIVES[args.objective], "game": "+".join(args.game)} | tags)
        mlflow.log_params(
            {
                "model": args.model,
                "game": ",".join(args.game),
                "held_out": ",".join(held_out),
                "batch_infosets": args.batch_infosets,
                "objective": args.objective,
                "steps": args.steps,
                "lr": args.lr,
                "lora_rank": args.lora_rank,
                "lora_targets": ",".join(LORA_TARGETS),
                "micro_batch": args.micro_batch,
                "mass_weight": args.mass_weight,
                "trainable_params": trainable,
                "shuffle_train": not args.no_shuffle,
                "cfr_iterations": args.cfr_iterations,
                "seed": args.seed,
                "relabel": args.relabel,
                "relabel_flip": args.relabel_flip,
                "adapter_dir": str(out),
                **(
                    {
                        "beta": args.beta,
                        "magnet": args.magnet,
                        "update": args.update,
                        "neurd_floor": args.neurd_floor,
                        "value_refresh": args.value_refresh,
                    }
                    if value_objective
                    else {}
                ),
                **({"window": args.window} if history is not None else {}),
            }
        )
        log.info("MLflow run %s started", mlflow.active_run().info.run_id)

    def run_eval(step: int) -> dict[str, tuple[PolicyResult, Evaluation]]:
        backend.model.eval()
        results = {name: evaluate_game(backend, game) for name, game in games.items()}
        backend.model.train()
        metrics: dict[str, float] = {}
        for name, (result, ev) in results.items():
            metrics |= eval_metrics(name, games[name], result, ev)
        if mlflow is not None:
            mlflow.log_metrics(metrics, step=step)
        log.info(
            "step %4d  eval  %s",
            step,
            "  ".join(
                f"{n} {100 * metrics[f'{n}/exploitability_norm']:.2f}% "
                f"(EV {metrics[f'{n}/seat0_ev_vs_eq']:+.3f}/{metrics[f'{n}/seat1_ev_vs_eq']:+.3f})"
                for n in games
            ),
        )
        for name in games:
            mass = metrics[f"{name}/legal_mass_min"]
            if mass < MIN_LEGAL_MASS:
                log.warning(
                    "step %4d  %s: legal mass %.3f < %.2f, the model gives invalid answers",
                    step,
                    name,
                    mass,
                    MIN_LEGAL_MASS,
                )
        return results

    relabel = (
        Relabeler(args.relabel, args.relabel_flip, np.random.default_rng(args.seed + 1))
        if args.relabel
        else None
    )
    try:
        start = time.perf_counter()
        results = run_eval(0)
        backend.model.train()
        for step in range(1, args.steps + 1):
            items = sample_items(train, args.batch_infosets, rng)
            common = {
                "shuffle": not args.no_shuffle,
                "seed": step,
                "micro_batch": args.micro_batch,
                "mass_weight": args.mass_weight,
            }
            if value_objective:
                stats = value_step(
                    backend,
                    items,
                    optimizer,
                    opponent="cfr" if args.objective == "value-vs-cfr" else "self",
                    beta=args.beta,
                    magnet=args.magnet,
                    update=args.update,
                    floor=args.neurd_floor,
                    history=history,
                    cache=value_cache,
                    refresh=(step - 1) % args.value_refresh == 0,
                    **common,
                )
            else:
                stats = distill_step(backend, items, optimizer, relabel=relabel, **common)
            if mlflow is not None:
                mlflow.log_metrics({f"train/{k}": v for k, v in stats.items()}, step=step)
            if step % args.eval_every == 0 or step == args.steps:
                log.info("step %4d  %s", step, "  ".join(f"{k} {v:.4f}" for k, v in stats.items()))
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
