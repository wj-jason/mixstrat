# Post-training design: teaching an LLM to reason about mixed-strategy poker

Status: draft for discussion. Nothing here is built yet.

## Goal

Post-train a small local LLM on 3-card Kuhn poker so that it

1. plays closer to equilibrium (lower exploitability) on the training game,
2. reasons more concisely and decisively, and
3. **transfers** that skill to variants it was never trained on.

The research question is (3): does RL on one game teach reasoning about card
strength, bluffing and calling ranges, or only memorized answers?

## Where we are (baselines, 3-card Kuhn)

All numbers from `mixstrat-eval`, logged in MLflow. Exploitability is the
chips per hand a perfect counter-strategy wins beyond the game value, as a
share of the 2-chip starting pot; 0 is equilibrium.

| Model | Mode | Exploitability | Notes |
|---|---|---|---|
| uniform random | - | 22.9% | reference |
| qwen3:1.7b | direct | 16.9% | bets and calls everything |
| qwen3:1.7b | reasoning, 1024 tokens | 22.3% | reasons badly: confuses card order, miscounts chips |
| qwen3:4b | direct | 16.6% | same habit as 1.7b |
| qwen3:4b | reasoning, 3000 tokens | 9.8% | 52% of traces cut off |
| qwen3:4b | reasoning, 6000 tokens | **7.0%** | 10% cut off, ~3400 tokens per decision |

What the best run (4b, 6000 tokens) still gets wrong is the middle card: it bets
Q (equilibrium checks) and always calls with Q (equilibrium calls 0.56 / 0.33).

**Reasoning style.** Traces average ~46 self-corrections ("wait", "hmm",
"alternatively") but change their tentative conclusion only ~1.3 times.
Self-corrections correlate 0.95 with length, and traces that change their mind
more are barely worse (distance from the equilibrium answer 0.31 vs 0.29). The
hedging mostly costs tokens, not accuracy, and token cost is what makes
reasoning evaluation and training slow on this GPU.

**Targets for transfer** (untrained baselines):

| Game | What changes | Untrained 1.7b direct |
|---|---|---|
| kuhn3 lowball | lowest card wins; equilibrium is Kuhn with J and K swapped | 17.1% |
| kuhn4 | extra rank; the two middle cards get different roles | 14.4% |

Lowball is the clean first test: a model that memorized "bet K, fold J" plays
it exactly backwards, while one that reads the rules transfers perfectly.

## Design

### 1. Training backend

Ollama serves quantized GGUF models and cannot train. Training loads the same
Qwen3 weights in HF `transformers` with LoRA adapters (`peft`):

| Model | Weights | Fits 8 GB for training? |
|---|---|---|
| Qwen3-1.7B | bf16 LoRA | yes, with room for activations |
| Qwen3-4B | 4-bit QLoRA (`bitsandbytes`) | probably, with gradient checkpointing; long reasoning sequences are the risk |

An `HFBackend` implements the existing `PolicyBackend` protocol (same prompts,
same `Answer: **` prefill, same letter readout), so `query_profile`,
`evaluate`, the HTML reports and MLflow tracking work on training checkpoints
unchanged. Ollama must be unloaded during training (`keep_alive: 0`); both
won't fit at once. Final checkpoints can be merged, converted to GGUF and
loaded into Ollama with `ollama create` for side-by-side evaluation.

### 2. Reward: exact action values from the game tree

Because the game tree is known, a trace's answer can be scored exactly instead
of by playing out noisy hands.

For an infoset `I` (player `i`, card `c`, betting node `n`) and a fixed
strategy profile `σ` for everything else, the conditional value of action `a` is

```
q(I, a) = Σ_o P(o | c) · reach_opp(n, o) · V(child_a | c, o)  /  Σ_o P(o | c) · reach_opp(n, o)
```

where `o` ranges over the opponent's cards, `reach_opp` is the opponent's
probability of reaching `n` under `σ`, and `V` is the chip value to player `i`
of continuing from the child under `σ` (the player's own later decisions use
the current policy). These are the same per-action values `best_response.py`
and `cfr.py` already compute.

A trace whose answer distribution is `p` (read from the letter logprobs) earns

```
R(trace) = Σ_a p_a · q(I, a)
```

This has no sampling noise from the deal or the opponent, and the answer part
is differentiable: the letter softmax gives exact gradients for the readout,
and only the reasoning tokens need a policy-gradient estimate.

### 3. Who the opponent is

| Option | Pros | Cons | Use |
|---|---|---|---|
| A. Imitate CFR: reward `-KL(σ_CFR(I) ‖ p)` | stable, fast, clear target | teaches answers, not reasoning; likely memorizes kuhn3 and transfers poorly; Player 0's equilibrium isn't unique | pipeline check only |
| B. Fixed CFR opponent: `q` computed against `σ_CFR` | stable, simple | the best reply to an equilibrium is often a *pure* strategy, so the model learns to exploit, not to mix | stage 1 |
| C. Self-play: `q` computed against the model's own current profile | targets the equilibrium; rewards reasoning about the opponent | plain policy-gradient self-play cycles in mixed games | stage 2+ |

For C, the current profile comes for free from the training batch: the K
traces per infoset are samples from the current policy. Two stabilizers:

- **Regularization** toward a reference policy (KL penalty to the frozen
  starting model, in the spirit of magnetic mirror descent), which makes
  regularized self-play converge in two-player zero-sum games instead of
  cycling.
- **Averaged opponents**: compute `q` against a reach-weighted average of
  recent profiles, not just the latest one.

Transfer is the goal, so the main runs should use C. A and B exist to get the
pipeline working and to give reference points.

### 4. Algorithm

**Direct mode (no reasoning).** One forward pass per infoset. The loss is the
exact expected value plus a KL anchor:

```
L = -Σ_I w_I Σ_a p_θ(a | I) q(I, a)  +  β · Σ_I w_I KL(p_θ(· | I) ‖ p_ref(· | I))
```

with `w_I` the infoset's reach probability under current play (or uniform).
There's no rollout cost; one step over kuhn3 is 12 forward passes.

**Reasoning mode (GRPO-style).** For each infoset, sample a group of K traces
from the current policy (thinking on, with a token budget), read each trace's
answer distribution `p` exactly, and score it:

```
R'(trace) = R(trace) - λ · tokens / 1000
A(trace)  = R'(trace) - mean over the group
```

Update with the policy-gradient loss `-A · log π_θ(trace tokens)` (clipped,
GRPO/PPO style) plus the exact readout gradient and the KL anchor. The length
penalty `λ` targets the hedging directly; `R` is on the scale of about ±2 chips,
so `λ ≈ 0.01-0.05` per 1000 tokens is a reasonable starting range to sweep. An
alternative is a hard budget that tightens over training (e.g. 3000 → 1500).

A minimal custom loop is simplest for direct mode. For reasoning mode, TRL's
GRPO trainer is worth evaluating against a custom loop, since the reward needs
infoset context and per-step access to the current profile.

### 5. Evaluation and logging

- Every N steps: evaluate the checkpoint on kuhn3 (learning) and kuhn3 lowball
  plus kuhn4 (transfer) through `HFBackend`, using the existing pipeline.
- Log to MLflow in a new `post-training` experiment: training curves (reward,
  mean trace length, length penalty, KL to reference), plus the evaluation
  metrics and the existing `report.html` / `reasoning.html` artifacts at each
  evaluation step, so reasoning traces can be compared across checkpoints.
- Headline metric: **lowball exploitability at the end of kuhn3 training**,
  compared with the untrained model (positive transfer) and with a model
  trained directly on lowball (the upper bound).

## Plan

| Stage | Setup | Purpose | Exit criteria |
|---|---|---|---|
| 0. Infra | `HFBackend`; verify HF and Ollama give matching letter distributions on the same prompts | trust the new backend | per-infoset distributions agree within quantization noise |
| 1a. Pipeline check | 1.7B, direct, reward A (imitate CFR) | trainer, rewards and eval loop work end to end | kuhn3 exploitability falls well below 16.9% within an hour |
| 1b. First real RL | 1.7B, direct, reward B then C | can exact-value rewards move the policy; does self-play stay stable | seat EVs vs CFR approach the game value (B); exploitability falls and doesn't oscillate (C) |
| 2. Reasoning RL | 4B QLoRA, reasoning, reward C, budget 1000-1500 tokens + length penalty | the actual question: concise *and* good reasoning | exploitability below the 7.0% baseline at a fraction of the 3400-token cost |
| 3. Transfer | freeze stage 1b and stage 2 models; evaluate on lowball and kuhn4 | does it generalize | lowball exploitability below the untrained baseline, compared with a lowball-trained upper bound |

## Compute estimates (RTX 5060, 8 GB)

| Stage | Cost per training step | Notes |
|---|---|---|
| 1 (direct) | seconds | 12 forward passes, no generation |
| 2 (reasoning) | ~15-30 min at K=8 and ~3400 tokens | 12 × 8 × 3400 ≈ 330k generated tokens per step with HF generation |

Stage 2 at full reasoning length would take days for a few hundred steps, which
is why it starts with a reduced budget and a length penalty. Shorter traces
are the goal anyway, and they make each step proportionally cheaper.

## Risks and open questions

- **Mixed strategies under RL.** Self-play policy gradient can cycle or
  collapse to pure strategies. Regularization and averaged opponents mitigate
  this; how well they work with LLM policies is part of the experiment.
- **Imitation vs reasoning.** Reward A could show fast "progress" that is pure
  memorization. Transfer evaluation is the guard; reward A is not used for the
  main runs.
- **4B memory.** QLoRA plus long reasoning sequences may not fit in 8 GB even
  with gradient checkpointing; fall back to shorter budgets or the 1.7B.
- **Generation speed.** HF `generate` is slow for long traces. Faster engines
  (e.g. vLLM) may help but haven't been tested in this WSL setup.
- **Readout drift.** Training could change how the model formats answers.
  `legal_mass` is already logged and should stay near 1.
- **kuhn4 card names.** With 4 ranks the prompts currently name cards 1-4, not
  J/Q/K/A. Decide before the transfer test whether kuhn4 should also change
  the card names.
- **Sample noise.** Evaluation with 4 traces per infoset moves in steps of
  0.25; checkpoint comparisons in reasoning mode want 8-16.

## Decisions needed

1. Reward for the main runs: self-play (C) as proposed, or stay against a
   fixed CFR opponent (B) longer?
2. Model for stage 2: 4B QLoRA (reasons better, tight on memory) or 1.7B
   (faster, but reasons poorly today)?
3. Add the kuhn3 lowball and kuhn4 presets now (with J/Q/K/A names for kuhn4)
   so the untrained baselines are logged before any training.
