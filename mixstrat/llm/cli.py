"""Command line: evaluate an Ollama model on one game, optionally logging to MLflow.

Needs the ``ollama`` extra, and ``tracking`` for --track:

    mixstrat-eval --game kuhn13 --model qwen3:1.7b --track
    mixstrat-eval --game kuhn_classic --mode reasoning --samples 4 --track --show-thinking
    python -m mixstrat.llm.cli --game kuhn13 --shuffle
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING

from mixstrat.game.cfr import solve
from mixstrat.game.config import load_games
from mixstrat.game.tree import GameTree
from mixstrat.llm.evaluate import evaluate, infoset_rows
from mixstrat.llm.policy import ChoiceCache, query_profile
from mixstrat.llm.prompts import DEFAULT_PREFILL
from mixstrat.llm.report import card_ranges, format_ranges, node_title

if TYPE_CHECKING:
    from mixstrat.llm.hf import HFBackend
    from mixstrat.llm.ollama import OllamaBackend
    from mixstrat.llm.tracking import RunLogger

PRESETS = Path(__file__).resolve().parents[2] / "configs" / "games.toml"


def _setup_logging(level: int) -> None:
    """Live progress from the ``mixstrat`` loggers to stderr (results go to stdout)."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("mixstrat")
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate an LLM's strategy on a game.")
    parser.add_argument("--game", default="kuhn13", help="preset name in --games")
    parser.add_argument("--games", type=Path, default=PRESETS)
    parser.add_argument(
        "--backend",
        choices=("ollama", "hf"),
        default="ollama",
        help="hf: Hugging Face weights (direct mode, no cache), e.g. to score a trained adapter",
    )
    parser.add_argument(
        "--model", default=None, help="default: qwen3:1.7b (ollama) or Qwen/Qwen3-1.7B (hf)"
    )
    parser.add_argument("--adapter", default=None, help="LoRA adapter directory (hf backend)")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--prefill", default=None, help='assistant prefix (default "Answer: **")')
    parser.add_argument("--mode", choices=("direct", "reasoning"), default="direct")
    parser.add_argument("--samples", type=int, default=4, help="reasoning traces per infoset")
    parser.add_argument(
        "--think-tokens", type=int, default=1024, help="token budget per reasoning trace"
    )
    parser.add_argument(
        "--num-ctx", type=int, default=None, help="Ollama context window (default 4096)"
    )
    parser.add_argument("--shuffle", action="store_true", help="shuffle answer letters")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfr-iterations", type=int, default=2000)
    parser.add_argument("--cache", type=Path, default=Path("outputs/llm_cache.jsonl"))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--track", action="store_true", help="log the run to MLflow")
    parser.add_argument("--experiment", default=None, help="default: llm-<mode>")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument(
        "--show-thinking", action="store_true", help="print each reasoning trace as it finishes"
    )
    parser.add_argument("--quiet", action="store_true", help="no live progress, just results")
    args = parser.parse_args(argv)
    _setup_logging(
        logging.WARNING if args.quiet else logging.DEBUG if args.show_thinking else logging.INFO
    )

    prefill = DEFAULT_PREFILL if args.prefill is None else args.prefill
    if args.backend == "hf" and args.mode != "direct":
        parser.error("the hf backend supports --mode direct only")
    if args.adapter is not None and args.backend != "hf":
        parser.error("--adapter needs --backend hf")

    games = load_games(args.games)
    if args.game not in games:
        parser.error(f"unknown game {args.game!r}; choose from {', '.join(games)}")
    cfg = games[args.game]
    tree = GameTree(cfg)
    equilibrium = solve(tree, iterations=args.cfr_iterations).profile

    reasoning = args.mode == "reasoning"
    backend: OllamaBackend | HFBackend
    if args.backend == "hf":
        from mixstrat.llm.hf import DEFAULT_MODEL, HFBackend

        # No cache: an adapter directory's weights can change between runs.
        cache = None
        model = args.model or DEFAULT_MODEL
        backend = HFBackend.load(model, adapter=args.adapter, prefill=prefill)
    else:
        from mixstrat.llm.ollama import OllamaBackend

        cache = None if args.no_cache else ChoiceCache(args.cache)
        model = args.model or "qwen3:1.7b"
        backend = OllamaBackend(
            model,
            host=args.host,
            prefill=prefill,
            mode=args.mode,
            samples=args.samples,
            think_tokens=args.think_tokens,
            seed=args.seed,
            num_ctx=args.num_ctx,
        )
    params = {
        "backend": backend.name,
        "model": model,
        **({"adapter": args.adapter} if args.adapter is not None else {}),
        "mode": args.mode,
        **({"samples": args.samples, "think_tokens": args.think_tokens} if reasoning else {}),
        "prefill": prefill,
        "shuffle": args.shuffle,
        "seed": args.seed,
        "cfr_iterations": args.cfr_iterations,
        **({"num_ctx": args.num_ctx} if args.num_ctx is not None else {}),
    }
    tracking: AbstractContextManager[RunLogger | None] = nullcontext()
    if args.track:
        from mixstrat.llm.tracking import tracked_run

        tracking = tracked_run(
            tree=tree,
            params=params,
            experiment=args.experiment or f"llm-{args.mode}",
            run_name=f"{cfg.name}-{model}-{args.mode}",
            tracking_uri=args.tracking_uri,
        )

    with backend, tracking as run:
        if run is not None:
            print(f"MLflow run {run.run_id} started", file=sys.stderr)
        start = time.perf_counter()
        result = query_profile(
            tree,
            backend,
            shuffle=args.shuffle,
            seed=args.seed,
            reasoning=reasoning,
            cache=cache,
            progress=run.progress if run is not None else None,
        )
        elapsed = time.perf_counter() - start

        ev = evaluate(tree, result.profile, equilibrium)
        rows = infoset_rows(tree, result, equilibrium)
        pot = cfg.initial_pot
        print(
            f"{cfg.name} | {backend.name} | {len(result.queries)} infosets "
            f"({result.num_cached} cached) | {elapsed:.1f}s"
        )
        print("exploitability, % of starting pot:")
        print(f"  LLM        {100 * ev.llm.normalized(pot):7.2f}%")
        print(f"  uniform    {100 * ev.uniform.normalized(pot):7.2f}%")
        print(
            f"  CFR        {100 * ev.equilibrium.normalized(pot):7.2f}%  ({args.cfr_iterations} it)"
        )
        v = ev.equilibrium.value
        print("EV vs CFR, chips/hand (CFR in the same seat):")
        print(f"  Player 0   {ev.seat_ev[0]:+.4f}  ({v:+.4f})")
        print(f"  Player 1   {ev.seat_ev[1]:+.4f}  ({-v:+.4f})")
        print(
            f"mass on legal letters: min {result.min_legal_mass:.3f}, "
            f"mean {result.mean_legal_mass:.3f}"
        )
        if reasoning:
            print(
                f"reasoning: {len(result.samples)} traces, mean {result.mean_think_tokens:.0f} "
                f"tokens, {100 * result.truncated_rate:.0f}% cut off at {args.think_tokens}"
            )
        print("strategy by card range (mixed ranges show average action probabilities):")
        for nid in tree.decision_ids:
            print(f"  {node_title(tree, nid)}")
            print(f"    LLM  {format_ranges(card_ranges(tree, result.profile, nid))}")
            print(f"    eq   {format_ranges(card_ranges(tree, equilibrium, nid))}")

        if run is not None:
            run.finish(
                result=result,
                equilibrium=equilibrium,
                evaluation=ev,
                rows=rows,
                metrics={
                    "legal_mass_min": result.min_legal_mass,
                    "legal_mass_mean": result.mean_legal_mass,
                    "num_infosets": len(result.queries),
                    "num_cached": result.num_cached,
                    "query_seconds": elapsed,
                    **(
                        {
                            "truncated_rate": result.truncated_rate,
                            "mean_think_tokens": result.mean_think_tokens,
                        }
                        if reasoning
                        else {}
                    ),
                },
            )
            print(f"logged MLflow run {run.run_id}")


if __name__ == "__main__":
    main()
