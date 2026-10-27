"""Run a sweep from configs/sweeps.toml: one ``mixstrat-train`` run at a time.

    mixstrat-sweep selfplay_stability --dry-run    # list the runs and estimated time
    mixstrat-sweep selfplay_stability              # run them

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
    parser = argparse.ArgumentParser(description="Run a mixstrat-train sweep from sweeps.toml.")
    parser.add_argument("sweep", nargs="?", help="sweep name (omit to list sweeps)")
    parser.add_argument("--sweeps", type=Path, default=SWEEPS)
    parser.add_argument("--dry-run", action="store_true", help="list runs and estimated time")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args(argv)

    with open(args.sweeps, "rb") as f:
        sweeps = tomllib.load(f)
    if args.sweep is None:
        for name, table in sweeps.items():
            runs = expand(name, table)
            hours = len(runs) * table.get("est_minutes", 0) / 60
            print(f"{name}: {len(runs)} runs, ~{hours:.1f} h. {table.get('description', '')}")
        return
    if args.sweep not in sweeps:
        parser.error(f"unknown sweep {args.sweep!r}; choose from {', '.join(sweeps)}")

    table = sweeps[args.sweep]
    runs = expand(args.sweep, table)
    logs = Path("outputs/logs/sweeps") / args.sweep
    checkpoints = Path("outputs/checkpoints/sweeps") / args.sweep
    pending = [r for r in runs if not _finished(logs / f"{r.slug}.log")]
    est = table.get("est_minutes", 0)
    print(
        f"{args.sweep}: {len(runs)} runs, {len(runs) - len(pending)} already finished, "
        f"{len(pending)} to go, ~{len(pending) * est / 60:.1f} h at ~{est} min each"
    )
    if args.dry_run:
        for run in runs:
            state = "done" if run not in pending else "todo"
            print(f"  [{state}] {run.name}")
            print(f"         mixstrat-train {' '.join(train_args(run.options))}")
        return

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
            f"sweep={args.sweep}",
            "--tag",
            f"variant={run.variant or '-'}",
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
            if args.stop_on_error:
                break
        else:
            remaining = (time.time() - start) / i * (len(pending) - i) / 60
            print(f"    done in {minutes:.1f} min; ~{remaining:.0f} min left", flush=True)

    total = (time.time() - start) / 3600
    print(f"{args.sweep}: finished in {total:.1f} h; {len(failed)} failed")
    for name in failed:
        print(f"  failed: {name}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
