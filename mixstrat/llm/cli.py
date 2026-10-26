"""Command line: evaluate an Ollama model on one game, optionally logging to MLflow.

Needs the ``ollama`` extra, and ``tracking`` for --track:

    mixstrat-eval --game kuhn13 --model qwen3:1.7b --track
    python -m mixstrat.llm.cli --game kuhn13 --shuffle
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from mixstrat.game.cfr import solve
from mixstrat.game.config import load_games
from mixstrat.game.tree import GameTree
from mixstrat.llm.evaluate import evaluate, infoset_rows
from mixstrat.llm.policy import ChoiceCache, query_profile
from mixstrat.llm.report import card_ranges, format_ranges, node_title

PRESETS = Path(__file__).resolve().parents[2] / "configs" / "games.toml"


def _progress(done: int, total: int) -> None:
    print(f"\r  queried {done}/{total} infosets", end="" if done < total else "\n", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate an Ollama model's strategy on a game.")
    parser.add_argument("--game", default="kuhn13", help="preset name in --games")
    parser.add_argument("--games", type=Path, default=PRESETS)
    parser.add_argument("--model", default="qwen3:1.7b")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--prefill", default=None, help='assistant prefix (default "Answer: **")')
    parser.add_argument("--shuffle", action="store_true", help="shuffle answer letters")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfr-iterations", type=int, default=2000)
    parser.add_argument("--cache", type=Path, default=Path("outputs/llm_cache.jsonl"))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--track", action="store_true", help="log the run to MLflow")
    parser.add_argument("--experiment", default="llm-direct")
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args(argv)

    from mixstrat.llm.ollama import DEFAULT_PREFILL, OllamaBackend

    prefill = DEFAULT_PREFILL if args.prefill is None else args.prefill

    games = load_games(args.games)
    if args.game not in games:
        parser.error(f"unknown game {args.game!r}; choose from {', '.join(games)}")
    cfg = games[args.game]
    tree = GameTree(cfg)
    equilibrium = solve(tree, iterations=args.cfr_iterations).profile

    start = time.perf_counter()
    cache = None if args.no_cache else ChoiceCache(args.cache)
    with OllamaBackend(args.model, host=args.host, prefill=prefill) as backend:
        result = query_profile(
            tree, backend, shuffle=args.shuffle, seed=args.seed, cache=cache, progress=_progress
        )
        backend_name = backend.name
    elapsed = time.perf_counter() - start

    ev = evaluate(tree, result.profile, equilibrium)
    rows = infoset_rows(tree, result, equilibrium)
    pot = cfg.initial_pot
    print(
        f"{cfg.name} | {backend_name} | {len(result.queries)} infosets "
        f"({result.num_cached} cached) | {elapsed:.1f}s"
    )
    print("exploitability, % of starting pot:")
    print(f"  LLM        {100 * ev.llm.normalized(pot):7.2f}%")
    print(f"  uniform    {100 * ev.uniform.normalized(pot):7.2f}%")
    print(f"  CFR        {100 * ev.equilibrium.normalized(pot):7.2f}%  ({args.cfr_iterations} it)")
    v = ev.equilibrium.value
    print("EV vs CFR, chips/hand (CFR in the same seat):")
    print(f"  Player 0   {ev.seat_ev[0]:+.4f}  ({v:+.4f})")
    print(f"  Player 1   {ev.seat_ev[1]:+.4f}  ({-v:+.4f})")
    print(
        f"mass on legal letters: min {result.min_legal_mass:.3f}, mean {result.mean_legal_mass:.3f}"
    )
    print("strategy by card range (mixed ranges show average action probabilities):")
    for nid in tree.decision_ids:
        print(f"  {node_title(tree, nid)}")
        print(f"    LLM  {format_ranges(card_ranges(tree, result.profile, nid))}")
        print(f"    eq   {format_ranges(card_ranges(tree, equilibrium, nid))}")

    if args.track:
        from mixstrat.llm.tracking import log_run

        run_id = log_run(
            tree=tree,
            result=result,
            equilibrium=equilibrium,
            evaluation=ev,
            rows=rows,
            params={
                "backend": backend_name,
                "model": args.model,
                "mode": "direct",
                "prefill": prefill,
                "shuffle": args.shuffle,
                "seed": args.seed,
                "cfr_iterations": args.cfr_iterations,
            },
            metrics={
                "legal_mass_min": result.min_legal_mass,
                "legal_mass_mean": result.mean_legal_mass,
                "num_infosets": len(result.queries),
                "num_cached": result.num_cached,
                "query_seconds": elapsed,
            },
            experiment=args.experiment,
            run_name=f"{cfg.name}-{args.model}",
            tracking_uri=args.tracking_uri,
        )
        print(f"logged MLflow run {run_id}")


if __name__ == "__main__":
    main()
