# Experiment log

Running notes on getting LLMs to play mixed-strategy poker variants. Numbers
come from `mixstrat-eval` / `mixstrat-train`; MLflow run ids are given so each
result can be reopened (`mlflow ui --backend-store-uri sqlite:///mlflow.db`).

**Metric.** Exploitability: chips per hand a perfect counter-strategy wins
beyond the game value, as a share of the starting pot (2 chips). 0 is
equilibrium; uniform random play is the reference. EVs are against the CFR+
equilibrium, which itself gets -1/18 = -0.056 in seat 0 and +0.056 in seat 1 on
3-card Kuhn (acting first is a disadvantage).

**Setup.** RTX 5060 (8 GB) under WSL2, Ollama 0.35.1, MLflow 3.17, PyTorch 2.11
(CUDA 12.8), transformers 5.21, peft 0.21.

## 2026-10-26

### 0. Game code checks

- Fixed three bugs before starting: duplicate bet/raise sizes (or sizes that
  print the same, e.g. `0.1234567` / `0.1234568`) silently merged different
  nodes into one infoset; `Profile` clipped probabilities without
  renormalizing; the `kuhn_classic` description called player 0 "P1".
- CFR+ on 3-card Kuhn matches the published equilibrium (Wikipedia, "Kuhn
  poker") to within 0.0005 on every action: bluff J at $\alpha \in [0, 1/3]$,
  bet K at $3\alpha$, call Q at $\alpha + 1/3$; player 1 bluffs J 1/3, calls Q
  1/3. Game value -0.05556 = -1/18. $\alpha$ drifts across the equilibrium family
  with more iterations (0.223 at 2000, 0.233 at 100k), as expected.

### 1. Direct mode: reading the answer from logprobs

**Method.** Show the infoset as a chat prompt with lettered options, prefill the
reply up to the letter, and read the letter probabilities from the next-token
distribution. One call per infoset, exact probabilities, no sampling.

Observations:
- Without a prefill, qwen3 starts with `**` or "You have...", so the first
  token carries almost no probability on the letters (legal mass ~1e-6).
  Prefilling `Answer: **` fixes it (legal mass 1.000 in every run since).
- When Ollama continues a prefilled turn with `think: false` it drops Qwen3's
  empty thinking block, leaving the model "inside" its thinking. The prefill
  now closes the block itself.
- Ollama's logprobs are raw model probabilities: identical under different
  temperature / top-p settings.
- Ollama is not bit-deterministic: the same text can differ by a few hundredths
  at close decisions.

**qwen3:1.7b, direct.**

| Game | Exploitability | Uniform | Seat 0 EV | Seat 1 EV | Run |
|---|---|---|---|---|---|
| kuhn13 | 8.25% | 20.67% | -0.099 (CFR -0.064) | -0.027 (CFR +0.064) | `b2bf284f` |
| kuhn13, letters shuffled | 8.06% | | | | `26d856d3` |
| kuhn_classic | 16.85% | 22.92% | -0.111 | -0.111 | `17e6ba12` |

- Strategy: bets almost every hand and calls every bet ("maniac / calling
  station"). Shuffling the answer letters doesn't change it, so it isn't
  letter-position bias.
- Losing in seat 1, where equilibrium play wins: it throws away the positional
  edge by betting every time player 0 checks.

### 2. Reasoning mode

**Method.** Sample a reasoning trace with thinking on, under a "think briefly"
instruction and a token budget; feed the trace back inside the thinking block,
followed by the same prefill, and read the letter probabilities exactly. Average
over K traces per infoset (K = 4 here). Fed-back traces fully determine the
answer (a trace ending "fold, option A" gives A = 1.0), so the readout follows
the reasoning.

Observations about the traces:
- Qwen3's natural thinking is ~2000 tokens per decision; the brevity
  instruction shortens it to ~550-1400 but not to "a few sentences".
- At a hard cap the trace is cut off mid-thought and forced to answer.
- Letter mislabelling (reasoning concludes "bet", then says "A", which is check)
  happened in ~1 of 34 concluded traces. The readout softened it (0.59 bet).

**qwen3:1.7b, reasoning, 1024 tokens** (`b7d2842b`): **22.34%**, worse than
direct (16.85%) and close to uniform. 25% of traces cut off, mean 695 tokens.
Failure modes in the traces: loses track of which card is higher ("since I have
a King... I lose"), reads a bet as proof of strength, miscounts chips. Folds K
to a bet ~50% of the time.

### 3. Model size: qwen3:4b

- VRAM: 4.2 GB of the 8 GB was held by Windows apps with no model loaded;
  freeing it (browser and app hardware acceleration) left room for the 4b
  (3.2 GB at 4k context, 3.9 GB at 8k).

| qwen3:4b on kuhn_classic | Exploitability | Cut off | Mean trace | Seat 0 EV | Seat 1 EV | Run |
|---|---|---|---|---|---|---|
| direct | 16.61% | - | - | -0.130 | -0.112 | `574f45ab` |
| reasoning, 1024 tokens | 16.24% | 98% | 1021 | -0.113 | -0.104 | `73c7b36a` |
| reasoning, 3000 tokens | 9.81% | 52% | 2556 | -0.109 | -0.015 | `ce359862` |
| reasoning, 6000 tokens, 8k context | **7.03%** | 10% | 3409 | -0.083 | -0.014 | `6fb7d512` |

- Size alone doesn't help direct mode (16.6% vs 16.9%): the "bet and call
  everything" habit isn't a reasoning failure.
- With room to finish thinking, the 4b gets the basics right: folds J to a bet,
  value-bets K, bluffs J at roughly the right frequency (0.25 vs 0.33).
- The remaining leak is the middle card: it bets Q (equilibrium checks) and
  always calls with Q (equilibrium calls 0.56 / 0.33).
- At 3000 tokens all of player 0's opening traces were cut off, and those were
  the spots it still played badly; the opening decision needs the most thought.
- Seeding works: traces that finished under 3000 tokens were token-for-token
  identical in the 6000-token run.
- Hedging (6000-token run, 48 traces): median 46 self-corrections ("wait",
  "hmm", ...) per trace, but only ~1.3 actual changes of conclusion. Corrections
  correlate 0.95 with length; traces that change their mind more are barely
  worse (distance from the equilibrium answer 0.31 vs 0.29). The back-and-forth
  costs tokens, not accuracy.

### 4. Transfer targets

Untrained qwen3:1.7b direct, for later comparison:

| Variant | Change | Equilibrium | Untrained |
|---|---|---|---|
| kuhn_lowball | lowest card wins | Kuhn with J and K swapped | 17.1% |
| kuhn4 | extra rank | two middle cards get different roles | 14.4% |

Rejected: player 1 acting first (training already covers both seats);
pot-size bets on 3 cards (degenerate: player 0 never bets, game value 0).
Lowball is the cleanest test: a model that memorized "bet K, fold J" plays it
exactly backwards.

### 5. Post-training, stage 0: HF backend vs Ollama

- First comparison showed per-infoset gaps up to 0.5. Feeding Ollama the
  HF-rendered text in raw mode reproduced the HF numbers, so the cause was
  **prompt format**, not quantization: Ollama's qwen3 template appends
  ` /no_think` to the last user message and leaves a blank line after the
  system header. The HF backend now reproduces that exactly.
- After matching: exploitability agrees within 0.2-0.3 points (kuhn_classic
  16.67% vs 16.85%, kuhn13 8.51% vs 8.25%), mean per-infoset difference
  0.02-0.03, legal mass 1.000 on both. The rest is bf16 vs Q4 weights plus
  Ollama's nondeterminism. Rule: train and score checkpoints with the HF
  backend; use Ollama for quick untrained baselines.

### 6. Post-training, stage 1a: distilling CFR (qwen3:1.7b, direct, LoRA)

Setup: LoRA rank 16 on all attention and MLP projections (17.4M params), lr
1e-4, 150 steps, all 12 kuhn_classic infosets per step, answer letters shuffled
every step, kuhn_lowball evaluated but not trained.

- **Speed.** First attempt took ~95 s per step: full-vocabulary logits were
  computed for every position, and activations for all 12 prompts (~8 GB)
  overflowed the GPU into system RAM. Keeping logits only at the last position,
  gradient checkpointing and micro-batches of 4 brought it to ~1.5 s per step
  at a 5.2 GB peak.

**Run 1** (`fb83b58d`, cross-entropy on the renormalized letter distribution):

| Step | KL to CFR | kuhn_classic | kuhn_lowball (held out) |
|---|---|---|---|
| 0 | - | 16.7% | 17.0% |
| 10-80 | ~0.40-0.49 | 20-30% | 20-30% |
| 100 | 0.18 | 6.4% | 34.3% |
| 150 | 0.003 | **0.63%** | **53.2%** |

- Stuck answering ~50/50 everywhere for ~80 steps (KL ~0.43 is what uniform
  answers score), then broke out and converged.
- Near-perfect Kuhn, **negative transfer**: on lowball it plays the same
  strategy (folds J, now the best card; calls with K, now the worst), more than
  twice as exploitable as random. Imitation memorizes card names.
- Legal mass fell to **0.73** at player 1 / J / after a check: 25% of the
  probability went to `F` (likely "Fold", which isn't an option there).
  Cause: the renormalized loss depends only on the letter logits, so the
  gradient on every other logit is exactly zero and leaked mass is never
  penalized.

**Run 2** (`eac0dde1`, loss = renormalized cross-entropy $- \log(\text{legal mass})$,
i.e. cross-entropy over the full vocabulary):

| | Run 1 | Run 2 |
|---|---|---|
| min legal mass, final | 0.730 | **0.999** |
| min legal mass, worst step | 0.709 | 0.980 |
| final KL to CFR | 0.003 | 0.007 |
| kuhn_classic, final (last 4 evals) | 0.63% (0.6-3.6%) | 1.13% (0.8-2.3%) |
| kuhn_lowball, final | 53.2% | 52.5% |
| plateau broke at | ~step 85 | ~step 75 |

- Invalid answers gone; in-game play still at equilibrium within noise;
  transfer still negative, as expected for imitation.

### 7. Post-training, stage 1b: training on exact chip values (qwen3:1.7b, direct)

Reward per infoset: the counterfactual action values $q(I,a)$ of the model's
answer distribution, with the model in its own seat and the opponent either CFR
(1b-i) or the model itself (1b-ii).

Getting it to learn at all took four fixes:
- **Plain expected value stalls.** The logit gradient of $\sum_a p_a q_a$ is
  $p_a(q_a - \bar v)$, which vanishes as $p_a \to 0$, so an action the model has
  stopped taking (folding J, $p = 10^{-4}$) never comes back, and large-gradient
  infosets drag the shared LoRA weights for small ones. Switched to NeuRD:
  $-\sum_a \text{sg}(q_a - \bar q)\log p_a$.
- **NeuRD needs a threshold.** Unchecked it keeps pushing losing actions further
  toward zero probability forever (loss diving to -21 while the strategy didn't
  change in 150 steps). Added a floor: no push below $p = 10^{-3}$ (or above
  $1 - 10^{-3}$).
- **Learning rate.** At 1e-4 one Adam step drives every probability to 0 or 1
  (Adam's first step is about $\text{lr} \times \text{sign}(g)$ on all 17M
  weights). 1e-5 was too slow to break the early plateau; 3e-5 works.
- **bf16 logits are quantized.** The gap between the two answer letters' logits
  moved only in steps of 0.25 (probabilities kept landing on sigmoid(1),
  sigmoid(2), ...), which rounds mixed strategies (1/3 → 0.32 or 0.38) and
  swallows small updates. The answer logits are now computed in float32 via a
  custom autograd function (no fp32 weight copy). This also makes HF evaluation
  more precise than in stage 1a.

Results (150 steps, lr 3e-5, uniform magnet $\beta = 0.02$):

| | kuhn3 | kuhn3 EV vs CFR | lowball (held out) | Run |
|---|---|---|---|---|
| value vs CFR | 12.50% | -0.056 / +0.056 | 66.7% | `39b952d8` |
| self-play (final) | 7.37% (ranged 7.4-16.9%) | -0.056 / +0.053 | 65.6% (ranged 34-67%) | `cf525b35` |

- **vs CFR** converged by step 60 to a pure best response: folds J, checks Q,
  never bluffs. Its EV against CFR is exactly the game value in both seats, as
  every best response to an equilibrium's is; being pure, it's exploitable.
- **Self-play cycles** between pure strategies, as predicted for unregularized
  self-play; the magnet ($\beta = 0.02$, at most ~0.014 chips) was too weak.
  The final checkpoint is pure: checks every opening hand, never bluffs.
- **Transfer got worse than imitation.** Across every training run so far,
  lowball never improved at any checkpoint: its best values all come from
  before the model learned anything on kuhn3, and it worsens as kuhn3 improves.
  With "highest card wins" in every training prompt, "K good, J bad" solves the
  training problem perfectly; no optimizer setting changes which solution is
  optimal. Next: vary the rules in training (`train_distribution` sweep) and
  stabilize self-play (`selfplay_stability` sweep).

## Takeaways so far

1. Without reasoning, both model sizes play the same exploitable "always
   aggressive" strategy (~16-17% on kuhn3).
2. Reasoning helps only with enough room to finish: harmful at 1.7b / 1024
   tokens, 7.0% at 4b / 6000 tokens. Cut-off traces are the main failure.
3. Qwen3's reasoning is long mostly because of hedging, not because it changes
   its mind; shorter reasoning is a cost goal more than an accuracy goal.
4. Imitating CFR reaches equilibrium on the training game and transfers
   negatively. This is the baseline any reward-based method has to beat on
   lowball (52-53%) while staying well under the untrained 17% on kuhn3.
5. Train against the full-vocabulary distribution, or the model learns to put
   probability on invalid answers.

## Next

Stage 1b: replace imitation with exact chip-value rewards (fixed CFR opponent,
then regularized self-play) and see whether rewarding outcomes instead of
answers changes what transfers to lowball. See `post_training_design.md`.
