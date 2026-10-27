"""Log an LLM evaluation run to MLflow.

Runs go to ``--tracking-uri``, else ``MLFLOW_TRACKING_URI``, else a local
SQLite store in the working directory. Browse them with:

    mlflow ui --backend-store-uri sqlite:///mlflow.db

Each run records the game config and query settings as params, the
evaluation as metrics, and as artifacts: an HTML strategy report (metrics,
glossary, card-by-action matrices), a per-infoset table, every prompt shown to
the model, and a strategy plot. While the model is being queried the run shows
as running, with live progress metrics (``infosets_done``, and for reasoning
``live_mean_think_tokens`` / ``live_truncated_rate``). Reasoning runs add every
reasoning trace, as a readable page (``reasoning.html``) and a filterable table (``reasoning.json``).
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np

from mixstrat.game.strategy import Profile
from mixstrat.game.tree import GameTree
from mixstrat.llm.evaluate import Evaluation
from mixstrat.llm.plots import strategy_figure
from mixstrat.llm.policy import InfosetQuery, PolicyResult
from mixstrat.llm.prompts import card_names
from mixstrat.llm.report import render_reasoning, render_report

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def _git_commit() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def _reasoning_table(tree: GameTree, result: PolicyResult) -> dict[str, list[Any]]:
    names = card_names(tree.config)
    columns: dict[str, list[Any]] = {
        k: []
        for k in (
            "player",
            "card",
            "history",
            "sample",
            "answer",
            "tokens",
            "truncated",
            "thinking",
        )
    }
    for q in result.queries:
        s = q.infoset
        labels = [a.label for a in tree.actions_at(s)]
        for i, sample in enumerate(q.samples, 1):
            probs = q.action_probs(sample.probs)
            columns["player"].append(s.player)
            columns["card"].append("?" if s.card is None else names[s.card])
            columns["history"].append(" ".join(s.history) or "(start)")
            columns["sample"].append(i)
            columns["answer"].append(
                " / ".join(f"{a} {p:.2f}" for a, p in zip(labels, probs, strict=True))
            )
            columns["tokens"].append(sample.tokens)
            columns["truncated"].append(sample.truncated)
            columns["thinking"].append(sample.thinking)
    return columns


class RunLogger:
    """Logs to one open MLflow run: live progress while querying, results at the end."""

    def __init__(self, run_id: str, tree: GameTree, backend: str) -> None:
        self.run_id = run_id
        self.tree = tree
        self.backend = backend
        self._tokens: list[int] = []
        self._truncated: list[bool] = []

    def progress(self, done: int, total: int, query: InfosetQuery) -> None:
        """Live metrics, stepped by infosets done, so the run's charts fill in as it goes."""
        live = {"infosets_done": float(done), "infosets_total": float(total)}
        if query.samples:
            self._tokens += [s.tokens for s in query.samples]
            self._truncated += [s.truncated for s in query.samples]
            live["live_mean_think_tokens"] = float(np.mean(self._tokens))
            live["live_truncated_rate"] = float(np.mean(self._truncated))
        mlflow.log_metrics(live, step=done)

    def finish(
        self,
        *,
        result: PolicyResult,
        equilibrium: Profile,
        evaluation: Evaluation,
        rows: list[dict[str, Any]],
        metrics: dict[str, float],
        prefix: str = "",
    ) -> None:
        """Log the evaluation. ``prefix`` (e.g. "final/kuhn_lowball/") namespaces metric
        names and artifact paths, so one run can hold results for several games."""
        tree = self.tree
        all_metrics = evaluation.metrics(tree.config.initial_pot) | metrics
        mlflow.log_metrics({f"{prefix}{k}": v for k, v in all_metrics.items()})

        table = {
            col: [str(r[col]) if isinstance(r[col], list) else r[col] for r in rows]
            for col in rows[0]
        }
        mlflow.log_table(table, f"{prefix}infosets.json")
        prompts = [
            {
                "infoset": str(q.infoset),
                "messages": q.prompt.messages,
                "letters": list(q.prompt.letters),
                "action_order": list(q.prompt.order),
                "probs": [round(float(p), 6) for p in q.probs],
            }
            for q in result.queries
        ]
        mlflow.log_text("\n".join(json.dumps(p) for p in prompts) + "\n", f"{prefix}prompts.jsonl")

        report = render_report(tree, result, equilibrium, evaluation, rows, self.backend)
        mlflow.log_text(report, f"{prefix}report.html")

        if result.samples:
            page = render_reasoning(tree, result, equilibrium, self.backend)
            mlflow.log_text(page, f"{prefix}reasoning.html")
            mlflow.log_table(_reasoning_table(tree, result), f"{prefix}reasoning.json")

        fig = strategy_figure(tree, result.profile, equilibrium)
        mlflow.log_figure(fig, f"{prefix}strategy.png")
        plt.close(fig)


@contextmanager
def tracked_run(
    *,
    tree: GameTree,
    params: dict[str, Any],
    experiment: str,
    run_name: str,
    tracking_uri: str | None = None,
) -> Iterator[RunLogger]:
    """Open an MLflow run for one evaluation. The run shows as running in the UI until
    the block exits, and is marked failed if the block raises."""
    mlflow.set_tracking_uri(
        tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    )
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"game": tree.config.name, "git_commit": _git_commit()})
        mlflow.log_params(params)
        mlflow.log_params({f"game.{k}": str(v) for k, v in tree.config.to_dict().items()})
        yield RunLogger(run.info.run_id, tree, str(params["backend"]))
