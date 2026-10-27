"""Card-order probe: does a trained adapter read the rules, or recall card labels?

    mixstrat-probe                      # base model + every train_distribution adapter
    mixstrat-probe --adapter base outputs/checkpoints/sweeps/train_distribution/...
    mixstrat-probe --track              # one MLflow run per adapter

Evaluates each model on the same 3-card game under different labels and payoff
rules (the card-order presets in configs/games.toml). A model that reads the
rules does equally well on games that are the same up to relabelling; one that
recalls J/Q/K habits does well wherever the J/Q/K answers happen to be right
(kuhn_classic and kuhn_lowball_reversed) and badly elsewhere.
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch

from mixstrat.game.cfr import solve
from mixstrat.game.config import load_games
from mixstrat.game.tree import GameTree
from mixstrat.llm.evaluate import infoset_rows
from mixstrat.llm.hf import DEFAULT_MODEL, HFBackend
from mixstrat.train.common import PRESETS, Game, eval_metrics, evaluate_game

log = logging.getLogger("mixstrat.train.probe")

GAMES = [
    "kuhn_classic",
    "kuhn_classic_234",
    "kuhn_reversed",
    "kuhn_lowball",
    "kuhn_lowball_234",
    "kuhn_lowball_reversed",
]
SWEEP_ADAPTERS = Path("outputs/checkpoints/sweeps/train_distribution")


def _label(adapter: str) -> str:
    return "base" if adapter == "base" else Path(adapter).name.removeprefix("train_distribution__")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--adapter",
        nargs="+",
        default=None,
        help='adapter directories; "base" is the model without one '
        f"(default: base + every adapter in {SWEEP_ADAPTERS})",
    )
    parser.add_argument("--game", nargs="+", default=GAMES)
    parser.add_argument("--games", type=Path, default=PRESETS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cfr-iterations", type=int, default=2000)
    parser.add_argument("--track", action="store_true", help="log one MLflow run per adapter")
    parser.add_argument("--experiment", default="card-order-probe")
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    adapters = args.adapter or ["base", *sorted(str(p) for p in SWEEP_ADAPTERS.iterdir())]
    presets = load_games(args.games)
    for name in args.game:
        if name not in presets:
            parser.error(f"unknown game {name!r}; choose from {', '.join(presets)}")
    games = []
    for name in args.game:
        tree = GameTree(presets[name])
        games.append(Game(tree, solve(tree, iterations=args.cfr_iterations).profile, name))

    mlflow: Any = None  # the mlflow module when tracking
    if args.track:
        import mlflow as mlflow_module

        from mixstrat.llm.tracking import DEFAULT_TRACKING_URI, RunLogger

        mlflow = mlflow_module
        mlflow.set_tracking_uri(
            args.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
        )
        mlflow.set_experiment(args.experiment)

    table: dict[str, dict[str, tuple[float, float]]] = {}
    for adapter in adapters:
        label = _label(adapter)
        log.info("== %s", label)
        backend = HFBackend.load(args.model, adapter=None if adapter == "base" else adapter)
        if mlflow is not None:
            mlflow.start_run(run_name=f"probe/{label}")
            mlflow.set_tags({"adapter": adapter, "model": args.model})
        table[label] = {}
        for game in games:
            result, ev = evaluate_game(backend, game)
            metrics = eval_metrics(game.name, game, result, ev)
            expl = metrics[f"{game.name}/exploitability_norm"]
            mass = metrics[f"{game.name}/legal_mass_min"]
            table[label][game.name] = (expl, mass)
            log.info("  %-22s %6.2f%%  (legal mass min %.3f)", game.name, 100 * expl, mass)
            if mlflow is not None:
                run_id = mlflow.active_run().info.run_id
                RunLogger(run_id, game.tree, backend.name).finish(
                    result=result,
                    equilibrium=game.equilibrium,
                    evaluation=ev,
                    rows=infoset_rows(game.tree, result, game.equilibrium),
                    metrics={k.split("/", 1)[1]: v for k, v in metrics.items()},
                    prefix=f"{game.name}/",
                )
        if mlflow is not None:
            mlflow.end_run()
        del backend
        gc.collect()
        torch.cuda.empty_cache()

    width = max(len(label) for label in table)
    names = [g.name for g in games]
    print("\nexploitability (% of pot; * = legal mass below 0.95)")
    print(" " * width + "".join(f"{n:>24}" for n in names))
    for label, row in table.items():
        cells = "".join(f"{100 * row[n][0]:23.1f}{'*' if row[n][1] < 0.95 else ' '}" for n in names)
        print(f"{label:<{width}}{cells}")


if __name__ == "__main__":
    main()
