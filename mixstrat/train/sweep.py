"""Run sweeps from configs/sweeps.toml: one ``mixstrat-train`` run at a time.

    mixstrat-sweep all                             # every sweep, in file order
    mixstrat-sweep all --dry-run                   # list the runs and estimated time
    mixstrat-sweep selfplay_stability              # one sweep

A sweep with a ``tune`` table takes some options from an earlier sweep's results
instead of its own config: for each combination of ``keys`` it averages
``metric`` over each run's last ``last`` evaluations, then over seeds, and uses
the combination with the lowest score. That rewards ending low *and* staying
there, so a run still cycling scores badly. The choice is saved to
``outputs/logs/sweeps/<sweep>/tuning.json`` and reused when the sweep resumes.

Every run is logged to MLflow (experiment ``post-training``), named
``<sweep>/<variant>-<grid values>`` and tagged ``sweep=<sweep>``, so filtering
the experiment by that tag (``tags.sweep = "selfplay_stability"``) shows the
whole sweep side by side. Each run's output goes to
``outputs/logs/sweeps/<sweep>/<run>.log`` and its adapter to
``outputs/checkpoints/sweeps/<sweep>/<run>/``.

Resumable: a run whose log ends with "logged MLflow run" is finished and is
skipped, so after an interruption the same command picks up where it stopped.
Failed runs are reported and the sweep moves on (``--stop-on-error`` to halt).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
import tomllib
import urllib.request
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SWEEPS = Path(__file__).resolve().parents[2] / "configs" / "sweeps.toml"
DONE_MARKER = "logged MLflow run"


@dataclass(frozen=True)
class Run:
    name: str  # "<sweep>/<variant>-<key>=<value>..."
    variant: str
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def slug(self) -> str:
        return self.name.replace("/", "__")


def _format(value: Any) -> str:
    return "+".join(map(str, value)) if isinstance(value, list) else str(value)


def expand(sweep: str, table: dict[str, Any]) -> list[Run]:
    """Every run in a sweep: each variant crossed with every grid combination."""
    base = table.get("base", {})
    grid = table.get("grid", {})
    variants = table.get("variants") or {"": {}}
    keys = list(grid)
    runs = []
    for variant, variant_options in variants.items():
        for values in itertools.product(*(grid[k] for k in keys)):
            combo = dict(zip(keys, values, strict=True))
            label = "-".join(f"{k}={_format(v)}" for k, v in combo.items())
            name = f"{sweep}/{'-'.join(x for x in (variant, label) if x)}"
            runs.append(Run(name, variant, base | variant_options | combo))
    return runs


def train_args(options: dict[str, Any]) -> list[str]:
    """``mixstrat-train`` command-line arguments for a dict of options."""
    args: list[str] = []
    for key, value in options.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            args += [flag] if value else []
        elif isinstance(value, list):
            args += [flag, *map(str, value)]
        else:
            args += [flag, str(value)]
    return args


def choose_options(
    runs: Sequence[tuple[dict[str, Any], Sequence[float]]], keys: Sequence[str], last: int
) -> tuple[dict[str, Any], dict[tuple[Any, ...], float]]:
    """Pick the combination of ``keys`` with the lowest score.

    ``runs`` holds each finished run's options and its metric values in step
    order. A run's score is the mean of its last ``last`` values; a
    combination's score is the mean over its runs (e.g. seeds). Returns the
    best options and every combination's score.
    """
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for options, values in runs:
        if values:
            tail = values[-last:]
            groups[tuple(options[k] for k in keys)].append(sum(tail) / len(tail))
    scores = {combo: sum(v) / len(v) for combo, v in groups.items()}
    best = min(scores, key=scores.__getitem__)
    return dict(zip(keys, best, strict=True)), scores


def _tuned_options(
    sweep: str, tune: dict[str, Any], sweeps: dict[str, Any], logs: Path
) -> dict[str, Any]:
    """Options for ``sweep`` chosen from the results of ``tune["from"]`` (see module docs)."""
    saved = logs / "tuning.json"
    if saved.exists():
        chosen: dict[str, Any] = json.loads(saved.read_text())["chosen"]
        print(f"{sweep}: reusing tuned options {chosen} from {saved}")
        return chosen

    import mlflow

    from mixstrat.llm.tracking import DEFAULT_TRACKING_URI

    source, keys = tune["from"], list(tune["keys"])
    metric, last = tune["metric"], int(tune.get("last", 5))
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI)
    client = mlflow.MlflowClient()
    found = mlflow.search_runs(
        experiment_names=["post-training"],
        filter_string=f"tags.sweep = '{source}' and attributes.status = 'FINISHED'",
    )
    # Recover each run's option values with the types from the source sweep's config.
    types = {r.name: r.options for r in expand(source, sweeps[source])}
    runs = []
    for _, row in found.iterrows():
        options = types.get(row["tags.mlflow.runName"])
        if options is None:
            continue
        history = sorted(client.get_metric_history(row["run_id"], metric), key=lambda m: m.step)
        runs.append((options, [m.value for m in history]))
    if not runs:
        print(f"{sweep}: no finished {source} runs to tune from; using the configured options")
        return {}

    chosen, scores = choose_options(runs, keys, last)
    print(f"{sweep}: {metric}, mean of the last {last} evals over seeds, from {len(runs)} runs:")
    for combo, score in sorted(scores.items(), key=lambda kv: kv[1]):
        mark = "  <- chosen" if dict(zip(keys, combo, strict=True)) == chosen else ""
        print(f"    {dict(zip(keys, combo, strict=True))}: {100 * score:.2f}%{mark}")
    logs.mkdir(parents=True, exist_ok=True)
    report = {
        "from": source,
        "metric": metric,
        "last": last,
        "chosen": chosen,
        "scores": {json.dumps(dict(zip(keys, c, strict=True))): s for c, s in scores.items()},
    }
    saved.write_text(json.dumps(report, indent=2) + "\n")
    return chosen


def _finished(log: Path) -> bool:
    return log.exists() and DONE_MARKER in log.read_text(errors="replace")


def _ollama_models() -> list[str]:
    """Models Ollama has loaded on the GPU (empty if Ollama isn't running)."""
    try:
        with urllib.request.urlopen("http://localhost:11434/api/ps", timeout=2) as response:
            return [m["name"] for m in json.load(response).get("models", [])]
    except OSError:
        return []


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run mixstrat-train sweeps from sweeps.toml.")
    parser.add_argument(
        "sweeps", nargs="*", help='sweep names in order, or "all" (omit to list sweeps)'
    )
    parser.add_argument("--file", type=Path, default=SWEEPS, help="sweep definitions")
    parser.add_argument("--dry-run", action="store_true", help="list runs and estimated time")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args(argv)

    with open(args.file, "rb") as f:
        sweeps = tomllib.load(f)
    if not args.sweeps:
        for name, table in sweeps.items():
            runs = expand(name, table)
            hours = len(runs) * table.get("est_minutes", 0) / 60
            print(f"{name}: {len(runs)} runs, ~{hours:.1f} h. {table.get('description', '')}")
        return
    names = list(sweeps) if args.sweeps == ["all"] else args.sweeps
    for name in names:
        if name not in sweeps:
            parser.error(f"unknown sweep {name!r}; choose from {', '.join(sweeps)} or all")

    if args.dry_run:
        total = 0.0
        for name in names:
            total += _dry_run(name, sweeps)
        print(f"total: ~{total:.1f} h")
        return

    start = time.time()
    failed: list[str] = []
    for name in names:
        failed += _run_sweep(name, sweeps, stop_on_error=args.stop_on_error)
        if failed and args.stop_on_error:
            break
    print(f"all done in {(time.time() - start) / 3600:.1f} h; {len(failed)} failed")
    for run_name in failed:
        print(f"  failed: {run_name}")
    if failed:
        sys.exit(1)


def _dry_run(sweep: str, sweeps: dict[str, Any]) -> float:
    """Print a sweep's runs; return the estimated hours left."""
    table = sweeps[sweep]
    runs = expand(sweep, table)
    logs = Path("outputs/logs/sweeps") / sweep
    pending = [r for r in runs if not _finished(logs / f"{r.slug}.log")]
    est = table.get("est_minutes", 0)
    print(
        f"{sweep}: {len(runs)} runs, {len(runs) - len(pending)} already finished, "
        f"{len(pending)} to go, ~{len(pending) * est / 60:.1f} h at ~{est} min each"
    )
    if "tune" in table:
        tune = table["tune"]
        print(f"  {', '.join(tune['keys'])} will be chosen from {tune['from']} before it starts")
    for run in runs:
        state = "done" if run not in pending else "todo"
        print(f"  [{state}] {run.name}")
        print(f"         mixstrat-train {' '.join(train_args(run.options))}")
    return len(pending) * est / 60


def _run_sweep(sweep: str, sweeps: dict[str, Any], *, stop_on_error: bool) -> list[str]:
    """Run a sweep's unfinished runs; return the names of runs that failed."""
    table = sweeps[sweep]
    logs = Path("outputs/logs/sweeps") / sweep
    checkpoints = Path("outputs/checkpoints/sweeps") / sweep
    tuned = _tuned_options(sweep, table["tune"], sweeps, logs) if "tune" in table else {}
    runs = [Run(r.name, r.variant, r.options | tuned) for r in expand(sweep, table)]
    pending = [r for r in runs if not _finished(logs / f"{r.slug}.log")]
    est = table.get("est_minutes", 0)
    print(
        f"{sweep}: {len(runs)} runs, {len(runs) - len(pending)} already finished, "
        f"{len(pending)} to go, ~{len(pending) * est / 60:.1f} h at ~{est} min each",
        flush=True,
    )

    loaded = _ollama_models()
    if loaded:
        print(
            f"warning: Ollama has {', '.join(loaded)} loaded on the GPU; training shares the "
            "8 GB. Unload with: curl localhost:11434/api/generate -d "
            '\'{"model":"<name>","keep_alive":0}\''
        )

    logs.mkdir(parents=True, exist_ok=True)
    env = os.environ | {
        "PYTHONUNBUFFERED": "1",
        "MLFLOW_DISABLE_AGENT_HINT": "1",
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
    }
    failed: list[str] = []
    start = time.time()
    extra_tags = ["--tag", f"tuned_from={table['tune']['from']}"] if tuned else []
    for i, run in enumerate(pending, 1):
        command = [
            sys.executable,
            "-m",
            "mixstrat.train.cli",
            *train_args(run.options),
            "--track",
            "--run-name",
            run.name,
            "--tag",
            f"sweep={sweep}",
            "--tag",
            f"variant={run.variant or '-'}",
            *extra_tags,
            "--out",
            str(checkpoints / run.slug),
        ]
        log = logs / f"{run.slug}.log"
        print(f"[{i}/{len(pending)}] {run.name}  (log: {log})", flush=True)
        run_start = time.time()
        with open(log, "w") as out:
            out.write("$ " + " ".join(command) + "\n")
            out.flush()
            code = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT, env=env).returncode
        minutes = (time.time() - run_start) / 60
        if code != 0 or not _finished(log):
            failed.append(run.name)
            print(f"    FAILED (exit {code}) after {minutes:.1f} min; see {log}", flush=True)
            if stop_on_error:
                break
        else:
            remaining = (time.time() - start) / i * (len(pending) - i) / 60
            print(f"    done in {minutes:.1f} min; ~{remaining:.0f} min left", flush=True)
    print(f"{sweep}: finished in {(time.time() - start) / 3600:.1f} h; {len(failed)} failed")
    return failed


if __name__ == "__main__":
    main()
