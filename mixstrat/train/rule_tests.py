"""Does a trained model read the rules, or recall what card labels meant in training?

    mixstrat-rules collect      # GPU: strategies of the base model + every sweep adapter
    mixstrat-rules collect --adapter outputs/checkpoints/some_run
    mixstrat-rules analyze      # CPU: tests 1-3 from the saved strategies

Every probe is 3-card Kuhn under one of 3 label sets (J/Q/K; 2/3/4; the made-up
Vex/Dorn/Pim) x 2 rank listings (conventional or reversed) x 2 showdown rules
(highest or lowest wins): 12 games that are strategically identical up to
relabelling. ``collect`` also records each model on the games its labels come
from (kuhn_classic, kuhn13, kuhn13_lowball), with fixed and shuffled answer
letters.

Test 1 (which hypothesis fits). Each hypothesis predicts a strategy at every
infoset; the one with the lowest mean total variation (TV) to the model wins:
  rules     the equilibrium of the probe game
  position  the last card listed is best, whatever the showdown rule
  label     the model's own play for that card name in the training game with the
            same showdown rule (undefined for the made-up names)
  base      the untrained model's play on the same prompt

Test 2 (one change at a time). Reverse the rank listing, flip the showdown
rule, or rename the cards, and compare how far the model's strategy moves with
how far the equilibrium moves. sensitivity = model shift / equilibrium shift
(1 = follows that line of the prompt, 0 = ignores it); alignment = cosine
between the two changes (+1 = moves the right way). Renaming should move
nothing (equilibrium shift 0), so its raw model shift is reported.

Test 3 (labels with no meaning). Exploitability on the made-up names. A model
that recalls labels has nothing to recall there; one that reads rules is unaffected.
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

from mixstrat.game.cfr import solve
from mixstrat.game.config import GameConfig, Payoff, load_games
from mixstrat.game.strategy import Profile
from mixstrat.game.tree import GameTree, Infoset
from mixstrat.llm.evaluate import evaluate
from mixstrat.llm.prompts import card_names
from mixstrat.llm.relabel import NEUTRAL_PROBE

log = logging.getLogger("mixstrat.train.rule_tests")

LABEL_SETS = {"jqk": ("J", "Q", "K"), "234": ("2", "3", "4"), "neutral": NEUTRAL_PROBE}
ORDERS = ("asc", "rev")
PAYOFFS: tuple[Payoff, ...] = ("high", "low")
REFERENCES = ("kuhn_classic", "kuhn13", "kuhn13_lowball")
VARIANTS = ("fixed", "shuffled")
HYPOTHESES = ("rules", "position", "label", "base")
OUT = Path("outputs/rule_tests")
SWEEP_ADAPTERS = Path("outputs/checkpoints/sweeps/train_distribution")
PRESETS = Path(__file__).resolve().parents[2] / "configs" / "games.toml"

Key = tuple[int, int, tuple[str, ...]]  # (player, card, history)
Strategy = dict[Key, np.ndarray]


@functools.cache
def _preset_names(game: str) -> tuple[str, ...]:
    return card_names(load_games(PRESETS)[game])


@functools.cache
def probe_configs() -> dict[str, GameConfig]:
    out = {}
    for labels, names in LABEL_SETS.items():
        for order in ORDERS:
            listed = names if order == "asc" else tuple(reversed(names))
            for payoff in PAYOFFS:
                key = f"{labels}/{order}/{payoff}"
                out[key] = GameConfig(name=key.replace("/", "_"), card_names=listed, payoff=payoff)
    return out


def label_reference(labels: str, payoff: str) -> str | None:
    """The training game whose meaning for these labels the label hypothesis uses."""
    if labels == "neutral":
        return None
    if payoff == "low":
        return "kuhn13_lowball"
    return "kuhn_classic" if labels == "jqk" else "kuhn13"


def _key(s: Infoset) -> Key:
    assert s.card is not None
    return (s.player, s.card, s.history)


def tv(a: np.ndarray, b: np.ndarray) -> float:
    return 0.5 * float(np.abs(a - b).sum())


# ------------------------------------------------------------------ collect


def _label(adapter: str) -> str:
    return "base" if adapter == "base" else Path(adapter).name.removeprefix("train_distribution__")


def collect(args: argparse.Namespace) -> None:
    import gc

    import torch

    from mixstrat.llm.hf import HFBackend
    from mixstrat.train.common import model_profile

    presets = load_games(PRESETS)
    trees = {k: GameTree(c) for k, c in probe_configs().items()}
    trees |= {f"ref/{name}": GameTree(presets[name]) for name in REFERENCES}
    adapters = args.adapter or ["base", *sorted(str(p) for p in SWEEP_ADAPTERS.iterdir())]
    if "base" not in adapters and not _load(args.out).get("base"):
        adapters = ["base", *adapters]  # test 1's base hypothesis needs it

    data = _load(args.out)
    for adapter in adapters:
        label = _label(adapter)
        if label in data and not args.force:
            log.info("%s: already collected", label)
            continue
        log.info("== %s", label)
        backend = HFBackend.load(args.model, adapter=None if adapter == "base" else adapter)
        games: dict[str, dict[str, list]] = {}
        for key, tree in trees.items():
            games[key] = {}
            for variant in VARIANTS:
                profile = model_profile(
                    backend, tree, args.micro_batch, shuffle=variant == "shuffled", seed=1
                )
                games[key][variant] = [
                    [s.player, s.card, list(s.history), profile.at(s).round(6).tolist()]
                    for s in tree.infosets
                ]
        data[label] = {"adapter": adapter, "games": games}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(data))
        log.info("saved %s to %s", label, args.out)
        del backend
        gc.collect()
        torch.cuda.empty_cache()


def _load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


# ------------------------------------------------------------------ analyze


def strategies(entry: dict, variant: str) -> dict[str, Strategy]:
    """Saved strategies of one model, per game, keyed by (player, card, history)."""

    def probs(rows: dict[str, list]) -> list[np.ndarray]:
        if variant == "mean":
            return [
                (np.array(f[3]) + np.array(s[3])) / 2
                for f, s in zip(rows["fixed"], rows["shuffled"], strict=True)
            ]
        return [np.array(r[3]) for r in rows[variant]]

    out = {}
    for game, rows in entry["games"].items():
        first = rows["fixed"]
        out[game] = {(r[0], r[1], tuple(r[2])): p for r, p in zip(first, probs(rows), strict=True)}
    return out


def equilibria(cfr_iterations: int) -> tuple[dict[str, GameTree], dict[str, Strategy]]:
    trees, eq = {}, {}
    for key, cfg in probe_configs().items():
        tree = GameTree(cfg)
        profile = solve(tree, cfr_iterations).profile
        trees[key] = tree
        eq[key] = {_key(s): profile.at(s) for s in tree.infosets}
    return trees, eq


def predictions(
    key: str, model: dict[str, Strategy], base: dict[str, Strategy], eq: dict[str, Strategy]
) -> dict[str, Strategy | None]:
    """Each hypothesis's predicted strategy on probe ``key``."""
    labels, order, payoff = key.split("/")
    preds: dict[str, Strategy | None] = {
        "rules": eq[key],
        "position": eq[f"{labels}/{order}/high"],
        "base": base[key],
        "label": None,
    }
    ref = label_reference(labels, payoff)
    if ref is not None:
        names = probe_configs()[key].card_names
        ref_names = _preset_names(ref)
        assert names is not None
        played = model[f"ref/{ref}"]
        preds["label"] = {
            (p, i, h): played[(p, ref_names.index(names[i]), h)] for (p, i, h) in eq[key]
        }
    return preds


def hypothesis_fit(
    model: dict[str, Strategy], base: dict[str, Strategy], eq: dict[str, Strategy]
) -> dict[str, dict[str, float]]:
    """Mean TV from the model to each hypothesis, per probe."""
    out = {}
    for key in eq:
        preds = predictions(key, model, base, eq)
        out[key] = {
            h: float(np.mean([tv(model[key][k], pred[k]) for k in eq[key]]))
            for h, pred in preds.items()
            if pred is not None
        }
    return out


def _shift(a: Strategy, b: Strategy, match: Callable[[Key], Key]) -> tuple[float, np.ndarray]:
    """Mean TV between ``a`` and ``b`` (matched by ``match``) and the change vector."""
    keys = sorted(a)
    diffs = [b[match(k)] - a[k] for k in keys]
    return float(np.mean([0.5 * np.abs(d).sum() for d in diffs])), np.concatenate(diffs)


def _same(k: Key) -> Key:
    return k


def _reversed(k: Key) -> Key:
    return (k[0], 2 - k[1], k[2])  # same card name after reversing a 3-rank listing


def one_change(model: dict[str, Strategy], eq: dict[str, Strategy]) -> dict[str, dict[str, float]]:
    pairs: dict[str, list[tuple[str, str, Callable[[Key], Key]]]] = {
        "reverse listing": [
            (f"{lab}/asc/{pay}", f"{lab}/rev/{pay}", _reversed)
            for lab in LABEL_SETS
            for pay in PAYOFFS
        ],
        "flip showdown": [
            (f"{lab}/{order}/high", f"{lab}/{order}/low", _same)
            for lab in LABEL_SETS
            for order in ORDERS
        ],
        "rename": [
            (f"{a}/{order}/{pay}", f"{b}/{order}/{pay}", _same)
            for a, b in itertools.combinations(LABEL_SETS, 2)
            for order in ORDERS
            for pay in PAYOFFS
        ],
    }
    out = {}
    for change, ps in pairs.items():
        m_shift, e_shift, m_vec, e_vec = [], [], [], []
        for a, b, match in ps:
            ms, mv = _shift(model[a], model[b], match)
            es, ev = _shift(eq[a], eq[b], match)
            m_shift.append(ms)
            e_shift.append(es)
            m_vec.append(mv)
            e_vec.append(ev)
        mv, ev = np.concatenate(m_vec), np.concatenate(e_vec)
        norm = np.linalg.norm(mv) * np.linalg.norm(ev)
        out[change] = {
            "model_shift": float(np.mean(m_shift)),
            "eq_shift": float(np.mean(e_shift)),
            "sensitivity": float(np.mean(m_shift) / np.mean(e_shift))
            if np.mean(e_shift) > 1e-9
            else float("nan"),
            "alignment": float(mv @ ev / norm) if norm > 1e-12 else float("nan"),
        }
    return out


def exploitabilities(
    model: dict[str, Strategy], trees: dict[str, GameTree], eq: dict[str, Strategy]
) -> dict:
    """Exploitability (fraction of the pot) of the model on every probe."""
    out = {}
    for key, tree in trees.items():
        table = {Infoset(p, c, h): model[key][(p, c, h)] for (p, c, h) in eq[key]}
        eq_table = {Infoset(p, c, h): v for (p, c, h), v in eq[key].items()}
        ev = evaluate(
            tree, Profile.from_infosets(tree, table), Profile.from_infosets(tree, eq_table)
        )
        out[key] = ev.llm.normalized(tree.config.initial_pot)
    return out


def analyze(args: argparse.Namespace) -> None:
    data = _load(args.out)
    if "base" not in data:
        sys.exit(f"no base model in {args.out}; run `mixstrat-rules collect` first")
    trees, eq = equilibria(args.cfr_iterations)
    base = strategies(data["base"], args.variant)
    labels = [k for k in data if args.only is None or k in args.only or k == "base"]
    results = {}
    for label in labels:
        model = strategies(data[label], args.variant)
        results[label] = {
            "test1": hypothesis_fit(model, base, eq) if label != "base" else None,
            "test2": one_change(model, eq),
            "test3": exploitabilities(model, trees, eq),
        }
    report = render(results, args.variant)
    print(report)
    (args.out.parent / f"report_{args.variant}.txt").write_text(report)
    (args.out.parent / f"results_{args.variant}.json").write_text(json.dumps(results, indent=1))


def render(results: dict, variant: str) -> str:
    w = max(len(k) for k in results)
    lines = [f"answer letters: {variant}", ""]

    lines += [
        "TEST 1  mean TV to each hypothesis (lower = fits better), over the 7 probes with",
        "        J/Q/K or 2/3/4 labels other than J<Q<K highest-wins (kuhn_classic itself).",
        "        wins = probes where that hypothesis fits best.",
        f"{'':{w}}" + "".join(f"{h:>10}" for h in HYPOTHESES) + "    wins (r/p/l/b)",
    ]
    for label, r in results.items():
        if r["test1"] is None:
            continue
        fits = {k: v for k, v in r["test1"].items() if "label" in v and k != "jqk/asc/high"}
        means = {h: np.mean([v[h] for v in fits.values()]) for h in HYPOTHESES}
        wins = {h: 0 for h in HYPOTHESES}
        for v in fits.values():
            wins[min(v, key=v.get)] += 1
        lines.append(
            f"{label:{w}}"
            + "".join(f"{means[h]:10.3f}" for h in HYPOTHESES)
            + "    "
            + "/".join(str(wins[h]) for h in HYPOTHESES)
        )

    lines += [
        "",
        "TEST 1 detail  best hypothesis per probe (labels/listing/showdown)",
    ]
    probes = list(probe_configs())
    lines.append(f"{'':{w}}" + "".join(f"{p:>16}" for p in probes))
    for label, r in results.items():
        if r["test1"] is None:
            continue
        cells = []
        for p in probes:
            v = r["test1"][p]
            best = min(v, key=v.get)
            cells.append(f"{best} {v[best]:.2f}")
        lines.append(f"{label:{w}}" + "".join(f"{c:>16}" for c in cells))

    lines += [
        "",
        "TEST 2  sensitivity (model shift / equilibrium shift; 1 = follows the line) and",
        "        alignment (cosine with the equilibrium's change; +1 = right direction).",
        "        rename: raw model shift in TV (should be 0).",
        f"{'':{w}}{'listing sens':>14}{'align':>8}{'showdown sens':>15}{'align':>8}"
        f"{'rename shift':>14}",
    ]
    for label, r in results.items():
        t = r["test2"]
        lines.append(
            f"{label:{w}}{t['reverse listing']['sensitivity']:14.2f}"
            f"{t['reverse listing']['alignment']:8.2f}"
            f"{t['flip showdown']['sensitivity']:15.2f}{t['flip showdown']['alignment']:8.2f}"
            f"{t['rename']['model_shift']:14.3f}"
        )

    lines += [
        "",
        "TEST 3  exploitability (% of pot) per probe; the untrained model is ~17",
        f"{'':{w}}" + "".join(f"{p:>16}" for p in probes),
    ]
    for label, r in results.items():
        lines.append(f"{label:{w}}" + "".join(f"{100 * r['test3'][p]:16.1f}" for p in probes))
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect", help="query models on the probe and reference games (GPU)")
    c.add_argument(
        "--adapter",
        nargs="+",
        default=None,
        help=f'adapter directories, or "base" (default: base + every adapter in {SWEEP_ADAPTERS})',
    )
    c.add_argument("--model", default=None)
    c.add_argument("--micro-batch", type=int, default=8)
    c.add_argument("--force", action="store_true", help="re-collect models already saved")
    a = sub.add_parser("analyze", help="run tests 1-3 on the collected strategies")
    a.add_argument("--variant", choices=(*VARIANTS, "mean"), default="fixed")
    a.add_argument("--only", nargs="+", default=None, help="model labels to report")
    a.add_argument("--cfr-iterations", type=int, default=2000)
    for p in (c, a):
        p.add_argument("--out", type=Path, default=OUT / "strategies.json")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    if args.command == "collect":
        if args.model is None:
            from mixstrat.llm.hf import DEFAULT_MODEL

            args.model = DEFAULT_MODEL
        collect(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
