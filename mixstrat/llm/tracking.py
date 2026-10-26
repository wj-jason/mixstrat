"""Log an LLM evaluation run to MLflow.

Runs go to ``--tracking-uri``, else ``MLFLOW_TRACKING_URI``, else a local
SQLite store in the working directory. Browse them with:

    mlflow ui --backend-store-uri sqlite:///mlflow.db

Each run records the game config and query settings as params, the
evaluation as metrics, and as artifacts: an HTML strategy report (metrics,
glossary, card-by-action matrices), a per-infoset table, every prompt shown to
the model, and a strategy plot.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any

import matplotlib.pyplot as plt
import mlflow

from mixstrat.game.strategy import Profile
from mixstrat.game.tree import GameTree
from mixstrat.llm.evaluate import Evaluation
from mixstrat.llm.plots import strategy_figure
from mixstrat.llm.policy import PolicyResult
from mixstrat.llm.report import render_report

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


def log_run(
    *,
    tree: GameTree,
    result: PolicyResult,
    equilibrium: Profile,
    evaluation: Evaluation,
    rows: list[dict[str, Any]],
    params: dict[str, Any],
    metrics: dict[str, float],
    experiment: str,
    run_name: str,
    tracking_uri: str | None = None,
) -> str:
    """Log one evaluation; returns the MLflow run id."""
    cfg = tree.config
    mlflow.set_tracking_uri(
        tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    )
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"game": cfg.name, "git_commit": _git_commit()})
        mlflow.log_params(params)
        mlflow.log_params({f"game.{k}": str(v) for k, v in cfg.to_dict().items()})
        mlflow.log_metrics(evaluation.metrics(cfg.initial_pot) | metrics)

        table = {
            col: [str(r[col]) if isinstance(r[col], list) else r[col] for r in rows]
            for col in rows[0]
        }
        mlflow.log_table(table, "infosets.json")
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
        mlflow.log_text("\n".join(json.dumps(p) for p in prompts) + "\n", "prompts.jsonl")

        report = render_report(tree, result, equilibrium, evaluation, rows, str(params["backend"]))
        mlflow.log_text(report, "report.html")

        fig = strategy_figure(tree, result.profile, equilibrium)
        mlflow.log_figure(fig, "strategy.png")
        plt.close(fig)
        return run.info.run_id
