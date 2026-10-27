"""Readable summaries of a strategy: per-card action ranges and an HTML report.

The ranges are a compact text view for the terminal and for diffing runs; the
report shows the full matrices instead.

``card_ranges`` condenses one decision node into contiguous card ranges that
play alike, e.g. ``2-3 check 0.55 / bet 0.45 | 4-8 check | 9-A check 0.4 / bet 0.6``.
A range with one action at probability >= ``PURE`` shows just that action;
otherwise it lists the range's average probability for each action. Adjacent
cards join a range while every card stays within ``MERGE_TOL`` of the range's
average. For monotone strategies these are the thresholds: "check below 9,
bet 9 and up".

``render_report`` builds a self-contained HTML page (logged to MLflow as
``report.html``) with the headline metrics, a glossary, and a card-by-action
probability matrix for every decision node, LLM next to equilibrium.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from mixstrat.game.strategy import Profile
from mixstrat.game.tree import GameTree
from mixstrat.llm.evaluate import Evaluation
from mixstrat.llm.policy import PolicyResult
from mixstrat.llm.prompts import card_names

PURE = 0.95  # a range at or above this for one action shows as that action alone
MERGE_TOL = 0.1  # max gap between a card's probabilities and its range's average
SHOW_MIN = 0.01  # actions below this average are left out of a mixed range


@dataclass(frozen=True)
class CardRange:
    start: str  # first card name in the range
    end: str  # last card name in the range
    actions: tuple[str, ...]
    probs: tuple[float, ...]  # average over the range's cards, per action

    @property
    def mixed(self) -> bool:
        return max(self.probs) < PURE

    def __str__(self) -> str:
        span = self.start if self.start == self.end else f"{self.start}-{self.end}"
        if not self.mixed:
            return f"{span} {self.actions[int(np.argmax(self.probs))]}"
        parts = [
            f"{a} {p:.2f}" for a, p in zip(self.actions, self.probs, strict=True) if p >= SHOW_MIN
        ]
        return f"{span} {' / '.join(parts)}"


def node_cards(tree: GameTree, nid: int) -> tuple[npt.NDArray[np.intp], list[str]]:
    """Cards the actor at ``nid`` can hold, and their names ("?" if they can't see it)."""
    actor = tree.nodes[nid].actor
    assert actor is not None
    if not tree.config.informed[actor]:
        return np.array([0]), ["?"]
    names = card_names(tree.config)
    cards = np.flatnonzero(tree.card_marginals[actor] > 0)
    return cards, [names[c] for c in cards]


def card_ranges(tree: GameTree, profile: Profile, nid: int) -> list[CardRange]:
    cards, labels = node_cards(tree, nid)
    actions = tuple(a.label for a in tree.nodes[nid].actions)
    rows = profile.probs[nid][cards]
    groups: list[list[int]] = []  # indexes into cards, one list per range
    for i in range(len(cards)):
        if groups:
            candidate = rows[[*groups[-1], i]]
            if np.abs(candidate - candidate.mean(axis=0)).max() <= MERGE_TOL:
                groups[-1].append(i)
                continue
        groups.append([i])
    return [
        CardRange(
            labels[g[0]],
            labels[g[-1]],
            actions,
            tuple(float(p) for p in rows[g].mean(axis=0)),
        )
        for g in groups
    ]


def format_ranges(ranges: list[CardRange]) -> str:
    return " | ".join(str(r) for r in ranges)


def node_title(tree: GameTree, nid: int) -> str:
    node = tree.nodes[nid]
    history = " ".join(node.history_labels) or "(start)"
    facing = " (facing a bet)" if any(a.kind == "fold" for a in node.actions) else ""
    return f"Player {node.actor} | {history}{facing}"


# ---------------------------------------------------------------- HTML report

GLOSSARY: list[tuple[str, str]] = [
    (
        "exploitability",
        "Chips per hand a perfect counter-strategy wins from the LLM beyond the game's "
        "equilibrium value, averaged over both seats. 0 means unexploitable (a Nash "
        "equilibrium). The headline number.",
    ),
    (
        "exploitability_norm",
        "exploitability / starting pot, so variants with different stakes compare. "
        "Uniform random play and the CFR+ solution give the reference points.",
    ),
    ("nash_conv", "Sum of both seats' best-response gains = 2 x exploitability."),
    (
        "br_value_p0 / br_value_p1",
        "What a best responder earns per hand sitting in that seat against the LLM's other "
        "seat. Compare with game_value (seat 0) and -game_value (seat 1).",
    ),
    (
        "seat0/1_ev_vs_eq",
        "The LLM's chips per hand in that seat against a fixed equilibrium (CFR) opponent, "
        "one that plays solid but does not adapt. Compare with CFR's own EV in the same seat "
        "(game_value for Player 0, -game_value for Player 1): acting first is a disadvantage, "
        "so even perfect play can lose chips in seat 0. seat*_loss_vs_eq is the difference, "
        "and is smaller than exploitability because the opponent isn't exploiting.",
    ),
    ("self_play_value", "Seat 0's EV when the LLM plays both seats."),
    (
        "game_value",
        "Seat 0's EV when both seats play equilibrium. Negative means acting first is a "
        "disadvantage in this game.",
    ),
    (
        "legal_mass_min / _mean",
        "Probability the model put on the offered answer letters before renormalizing. "
        "Near 1 means the probabilities measure its choices; well below 1 means the prompt "
        "format broke and results are not trustworthy.",
    ),
    (
        "tv (per infoset)",
        "Total-variation distance between the LLM's and the equilibrium's action "
        "distributions: 0 = identical, 1 = no overlap. Equilibria need not be unique, so a "
        "large tv marks a difference worth a look, not necessarily a mistake.",
    ),
]

_CSS = """
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --border: rgba(11,11,11,0.10);
  --s1: #cde2fb; --s2: #9ec5f4; --s3: #6da7ec; --s4: #3987e5; --s5: #256abf; --s6: #184f95; --s7: #0d366b;
  --t1: #0b0b0b; --t2: #0b0b0b; --t3: #0b0b0b; --t4: #ffffff; --t5: #ffffff; --t6: #ffffff; --t7: #ffffff;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --border: rgba(255,255,255,0.10);
    --s1: #104281; --s2: #184f95; --s3: #1c5cab; --s4: #256abf; --s5: #3987e5; --s6: #6da7ec; --s7: #9ec5f4;
    --t1: #f0efec; --t2: #f0efec; --t3: #f0efec; --t4: #f0efec; --t5: #0b0b0b; --t6: #0b0b0b; --t7: #0b0b0b;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --border: rgba(255,255,255,0.10);
  --s1: #104281; --s2: #184f95; --s3: #1c5cab; --s4: #256abf; --s5: #3987e5; --s6: #6da7ec; --s7: #9ec5f4;
  --t1: #f0efec; --t2: #f0efec; --t3: #f0efec; --t4: #f0efec; --t5: #0b0b0b; --t6: #0b0b0b; --t7: #0b0b0b;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 32px 0 8px; }
h3 { font-size: 14px; margin: 0 0 6px; }
.sub { color: var(--ink-2); margin: 0 0 16px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 8px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; }
.tile .label { color: var(--ink-2); font-size: 12px; }
.tile .value { font-size: 22px; font-weight: 600; }
.tile .note { color: var(--muted); font-size: 12px; }
details { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px; }
summary { cursor: pointer; color: var(--ink-2); }
dl { display: grid; grid-template-columns: minmax(140px, max-content) 1fr; gap: 6px 16px; margin: 8px 0 0; }
dt { font-family: ui-monospace, monospace; font-size: 12px; color: var(--ink); }
dd { margin: 0; color: var(--ink-2); }
.node { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 12px; margin: 0 0 12px; }
.scroll { overflow-x: auto; }
table { border-collapse: separate; border-spacing: 2px; font-variant-numeric: tabular-nums; }
th { font-weight: 500; color: var(--ink-2); font-size: 12px; padding: 2px 4px; text-align: center; }
th.row { text-align: left; white-space: nowrap; padding-right: 8px; }
th.row span { color: var(--muted); font-weight: 400; }
tr.first th.row { padding-top: 6px; }
td { width: 40px; min-width: 34px; height: 26px; text-align: center; font-size: 12px;
  border-radius: 3px; color: var(--muted); }
td.b1 { background: var(--s1); color: var(--t1); } td.b2 { background: var(--s2); color: var(--t2); }
td.b3 { background: var(--s3); color: var(--t3); } td.b4 { background: var(--s4); color: var(--t4); }
td.b5 { background: var(--s5); color: var(--t5); } td.b6 { background: var(--s6); color: var(--t6); }
td.b7 { background: var(--s7); color: var(--t7); }
tr.gap td, tr.gap th { height: 6px; padding: 0; }
.legend { display: flex; gap: 2px; align-items: center; color: var(--ink-2); font-size: 12px; margin: 4px 0 12px; }
.legend span.sw { width: 28px; height: 14px; border-radius: 3px; display: inline-block; }
table.dev td { width: auto; text-align: left; padding: 2px 8px; color: var(--ink-2); }
table.dev th { text-align: left; }
"""


def _bin(p: float) -> int:
    """Sequential color bin 0 (none) to 7 for a probability."""
    if p < 0.005:
        return 0
    return min(7, 1 + int(p * 7))


def _matrix(tree: GameTree, nid: int, llm: Profile, eq: Profile) -> str:
    cards, labels = node_cards(tree, nid)
    node = tree.nodes[nid]
    who = "Player " + str(node.actor)
    history = " ".join(node.history_labels) or "the start"
    rows = ["<tr><th></th>" + "".join(f"<th>{html.escape(c)}</th>" for c in labels) + "</tr>"]
    for a, action in enumerate(node.actions):
        if a:
            rows.append(f'<tr class="gap"><td colspan="{len(cards) + 1}"></td></tr>')
        for source, profile, cls in (("LLM", llm, "first"), ("CFR", eq, "")):
            cells = []
            for card, label in zip(cards, labels, strict=True):
                p = float(profile.probs[nid][card, a])
                tip = f"{who} holding {label} after {history}: {action.label} {p:.3f} ({source})"
                cells.append(f'<td class="b{_bin(p)}" title="{html.escape(tip)}">{p:.2f}</td>')
            head = f"{html.escape(action.label)} <span>{source}</span>"
            rows.append(f'<tr class="{cls}"><th class="row">{head}</th>{"".join(cells)}</tr>')
    return '<div class="scroll"><table>' + "".join(rows) + "</table></div>"


def _tile(label: str, value: str, note: str = "") -> str:
    return (
        f'<div class="tile"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)}</div><div class="note">{html.escape(note)}</div></div>'
    )


def render_report(
    tree: GameTree,
    result: PolicyResult,
    equilibrium: Profile,
    evaluation: Evaluation,
    rows: list[dict[str, Any]],
    backend: str,
) -> str:
    cfg = tree.config
    pot = cfg.initial_pot
    ev = evaluation
    tiles = "".join(
        [
            _tile(
                "Exploitability (LLM)",
                f"{100 * ev.llm.normalized(pot):.2f}% pot",
                f"{ev.llm.exploitability:.4f} chips/hand",
            ),
            _tile("Uniform random", f"{100 * ev.uniform.normalized(pot):.2f}% pot", "reference"),
            _tile(
                "Equilibrium (CFR+)",
                f"{100 * ev.equilibrium.normalized(pot):.3f}% pot",
                "should be ~0",
            ),
            _tile(
                "Player 0 EV vs CFR",
                f"{ev.seat_ev[0]:+.4f}",
                f"chips/hand; CFR as Player 0: {ev.equilibrium.value:+.4f}",
            ),
            _tile(
                "Player 1 EV vs CFR",
                f"{ev.seat_ev[1]:+.4f}",
                f"chips/hand; CFR as Player 1: {-ev.equilibrium.value:+.4f}",
            ),
            _tile("Legal-letter mass", f"{result.min_legal_mass:.3f}", "minimum over infosets"),
        ]
    )
    glossary = "".join(f"<dt>{html.escape(k)}</dt><dd>{html.escape(v)}</dd>" for k, v in GLOSSARY)

    nodes = []
    for nid in tree.decision_ids:
        nodes.append(
            f'<section class="node"><h3>{html.escape(node_title(tree, nid))}</h3>'
            f"{_matrix(tree, nid, result.profile, equilibrium)}</section>"
        )

    worst = sorted(rows, key=lambda r: -r["tv"])[:10]
    dev_rows = "".join(
        f"<tr><td>P{r['player']}</td><td>{html.escape(str(r['card']))}</td>"
        f"<td>{html.escape(r['history'])}</td><td>{html.escape(r['actions'])}</td>"
        f"<td>{r['llm']}</td><td>{r['equilibrium']}</td><td>{r['tv']:.3f}</td></tr>"
        for r in worst
    )
    swatches = "".join(
        f'<span class="sw" style="background:var(--s{i})"></span>' for i in range(1, 8)
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Strategy report</title><style>{_CSS}</style></head>
<body><main>
<h1>{html.escape(cfg.name)}: strategy report</h1>
<p class="sub">{html.escape(backend)} &middot; {len(result.queries)} infosets &middot;
{html.escape(cfg.description or "")}</p>
<div class="tiles">{tiles}</div>
<h2>What the metrics mean</h2>
<details><summary>Glossary</summary><dl>{glossary}</dl></details>
<h2>Strategy by decision point</h2>
<p class="sub">Each grid shows P(action | card) as a probability from 0 to 1: for every action, the LLM's row
above the equilibrium's. Hover a cell for details.</p>
<div class="legend">0&nbsp;{swatches}&nbsp;1</div>
{"".join(nodes)}
<h2>Largest differences from equilibrium</h2>
<div class="scroll"><table class="dev"><tr><th>seat</th><th>card</th><th>history</th>
<th>actions</th><th>LLM</th><th>equilibrium</th><th>tv</th></tr>{dev_rows}</table></div>
</main></body></html>
"""


# ---------------------------------------------------------------- reasoning traces

_REASONING_CSS = """
.infoset { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 12px; margin: 0 0 12px; }
.dist { font-family: ui-monospace, monospace; font-size: 12px; color: var(--ink-2); margin: 2px 0; }
.dist { white-space: pre-wrap; }
.dist b { color: var(--ink); font-family: system-ui, sans-serif; font-weight: 600;
  display: inline-block; min-width: 92px; }
.infoset details { margin: 6px 0 0; padding: 6px 10px; }
.infoset summary { font-family: ui-monospace, monospace; font-size: 12px; }
.cut { color: var(--ink); font-weight: 600; }
pre { white-space: pre-wrap; word-break: break-word; font: 13px/1.5 system-ui, sans-serif;
  color: var(--ink); margin: 8px 0 0; }
"""


def _dist_text(labels: list[str], probs: Any) -> str:
    return " / ".join(f"{a} {float(p):.2f}" for a, p in zip(labels, probs, strict=True))


def render_reasoning(
    tree: GameTree, result: PolicyResult, equilibrium: Profile, backend: str
) -> str:
    """Every reasoning trace, grouped by infoset, with the answer each trace led to."""
    names = card_names(tree.config)
    sections = []
    for q in result.queries:
        s = q.infoset
        labels = [a.label for a in tree.actions_at(s)]
        card = "?" if s.card is None else names[s.card]
        title = f"{node_title(tree, tree.infoset_node[s])} | card {card}"
        traces = []
        for i, sample in enumerate(q.samples, 1):
            answer = _dist_text(labels, q.action_probs(sample.probs))
            cut = ' <span class="cut">cut off</span>' if sample.truncated else ""
            traces.append(
                f"<details><summary>sample {i}: {html.escape(answer)} &middot; "
                f"{sample.tokens} tokens{cut}</summary>"
                f"<pre>{html.escape(sample.thinking) or '(no thinking)'}</pre></details>"
            )
        sections.append(
            f'<section class="infoset"><h3>{html.escape(title)}</h3>'
            f'<div class="dist"><b>Options</b>{html.escape("   ".join(q.prompt.option_lines))}</div>'
            f'<div class="dist"><b>LLM average</b>{html.escape(_dist_text(labels, q.probs))}</div>'
            f'<div class="dist"><b>Equilibrium</b>'
            f"{html.escape(_dist_text(labels, equilibrium.at(s)))}</div>"
            f"{''.join(traces)}</section>"
        )
    samples = result.samples
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reasoning traces</title><style>{_CSS}{_REASONING_CSS}</style></head>
<body><main>
<h1>{html.escape(tree.config.name)}: reasoning traces</h1>
<p class="sub">{html.escape(backend)} &middot; {len(samples)} traces over {len(result.queries)}
infosets &middot; mean {result.mean_think_tokens:.0f} tokens &middot;
{100 * result.truncated_rate:.0f}% cut off at the token budget</p>
<p class="sub">Each trace is one sampled chain of thought; the answer next to it is the
model's letter distribution right after that trace. "LLM average" is the mean over traces,
the strategy that gets scored. Open a sample to read it.</p>
{"".join(sections)}
</main></body></html>
"""
