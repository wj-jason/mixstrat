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

## 2026-10-27

### 8. Sweeps: self-play stability and training distribution (qwen3:1.7b, direct)

`mixstrat-sweep all`, 24 runs, 5 h 47 m, none failed. Experiment `post-training`,
filter `tags.sweep`.

**`selfplay_stability`** ($\beta \in \{0.05, 0.1, 0.2, 0.5\}$ × opponent window
$\in \{1, 10\}$ × 2 seeds, 150 steps, kuhn3). kuhn3 exploitability, mean of the
last 5 evals and both seeds:

| $\beta$ \ window | 1 | 10 |
|---|---|---|
| 0.05 | 8.3% | 10.3% |
| 0.1 | 13.9% | 12.7% |
| 0.2 | 9.5% | 11.3% |
| 0.5 | 13.7% | 14.9% |

- Every run still cycles (6-21% through step 150); none settles. The first ~60
  steps are nearly identical across all 16 runs, so the cycling comes from the
  update itself, not the regularizer settings. The auto-tuner's choice
  ($\beta$ 0.05, window 1) is within run-to-run noise.

**`train_distribution`** (400 steps, batch 24, kuhn_lowball held out; A = kuhn3,
B = + kuhn13, C = + kuhn13_lowball, D = + kuhn13_low_heavy, kuhn13_duplicates).
kuhn_lowball exploitability (untrained 17.1%), last 5 evals / final:

| Variant | Distill | Self-play | Training games (distill) |
|---|---|---|---|
| A | 54.1 / 54.4 | 41.4 / 49.6 | ~0.5% |
| B | 53.4 / 53.3 | 56.5 / 53.1 | ~0.3% |
| C | 43.2 / 47.1 | 22.7 / 17.0 | ~0.2% |
| D | 42.9 / 39.9 | 46.3 / 66.6 | 1-3% |

- Distillation learns every training game, including 13-card lowball (0.1%),
  but 3-card lowball always ends up worse than untrained. More varied training
  only shrinks the damage (54% → 40%).
- C self-play dipped to 13.4% (step 225) and 7.2% (step 350, with legal mass
  0.88) but bounced back each time; that's cycling, not transfer.
- C distill on lowball: folds K to a bet (correct) but also folds J, the best
  card. The rule is applied half-way.

### 9. Does the model read the rules? Card-order probe and tests 1-3

`mixstrat-probe` and `mixstrat-rules` (`outputs/rule_tests/`). Probe games: 3-card
Kuhn under 3 label sets (J/Q/K, 2/3/4, the made-up Vex/Dorn/Pim) × rank list
in order or reversed × highest or lowest wins. All 12 are the same game up to
relabelling, so a model that reads the rules scores the same on all of them.

**Test 1: which hypothesis predicts the strategy.** Mean TV to each prediction
on the 7 probes whose labels have a training meaning (distill adapters):

| | rules | last listed best | label recall | untrained |
|---|---|---|---|---|
| A | 0.41 | 0.28 | **0.22** | 0.61 |
| B | 0.42 | 0.30 | **0.29** | 0.68 |
| C | 0.44 | 0.39 | **0.20** | 0.61 |
| D | 0.41 | 0.35 | **0.26** | 0.53 |

**Test 2: one prompt line changed at a time.** Sensitivity = model shift /
equilibrium shift; alignment = cosine between the two changes.

| | reverse list: sens / align | flip showdown: sens / align | rename shift (ideal 0) |
|---|---|---|---|
| base | 0.04 / 0.02 | 0.02 / 0.03 | 0.05 |
| A | 0.67 / -0.02 | 0.06 / -0.12 | 0.24 |
| B | 0.59 / -0.04 | 0.11 / -0.18 | 0.32 |
| C | 0.38 / 0.10 | 0.42 / 0.08 | 0.36 |
| D | 0.51 / 0.21 | 0.53 / 0.20 | 0.35 |

**Test 3: made-up labels.** A distill: 11.3% / 4.2% when highest wins (list in
order / reversed), 42.6% / 54.2% when lowest wins.

- Label recall fits best for every adapter, and the rules fit worst. Renaming,
  which should change nothing, moves trained models 5-7× more than the
  untrained one.
- Reversing the rank list changes behaviour a lot, but in no consistent
  direction (alignment ≈ 0). Models never trained on lowball ignore the
  showdown line.
- With labels that mean nothing, the model falls back on "last listed is
  best": right whenever highest wins, wrong whenever lowest wins.
- Same picture with shuffled answer letters. Caveats: one adapter per setting,
  12 infosets per probe; self-play adapters are near-constant and excluded.

So the model never works out card strength from the rules: it recalls what a
label meant in training, falls back on list position for new labels, and mostly
ignores the showdown rule. Every earlier training game used fixed labels, which
explains section 8.

### 10. Label-randomized distillation (`label_randomization` sweep)

`--relabel 0.75 --relabel-flip 0.5` (`mixstrat/llm/relabel.py`). 25% of training
prompts are unchanged; 37.5% get random card names (made-up words, numbers 5-99
in random order, Greek letters, colours, animals); 37.5% get random names plus a
flip (rank list reversed, showdown rule reversed). The target strategy is the
same in all three, since every view is the same game. Probe labels never appear,
and the J/Q/K prompts never say "lowest wins", so every held-out game is a new
prompt. All held-out games are still kuhn3 up to relabelling: this tests
recognizing the same game, not playing a new one.

kuhn3 only (variant A), 400 steps, otherwise as in section 8:

| Step | classic | lowball | lowball 2/3/4 | reversed | classic 2/3/4 | lowball reversed |
|---|---|---|---|---|---|---|
| seed 0, 225 | 23.5 | 27.8 | 27.0 | 25.0 | 24.9 | 27.0 |
| seed 0, 275 | 5.5 | 49.7 | 23.3 | 15.9 | 14.6 | 28.5 |
| seed 0, 325 | 5.0 | 5.9 | 4.4 | 3.0 | 2.9 | 7.2 |
| **seed 0, 400** (`e73145ac`) | **1.1** | **3.1** | **1.1** | **2.2** | **0.9** | **1.6** |
| seed 1, 400 (`12e1ce9b`) | 22.8 | 25.2 | 24.9 | 24.1 | 24.1 | 24.7 |

- **Seed 0 transfers to every held-out game,** including kuhn_lowball at 3.1%
  (every earlier run: 40-54%). It got there by overriding a conflicting prior:
  the unchanged prompts still teach "J is weakest", and kuhn_lowball says J is
  best.
- **Both seeds first sit on a card-blind plateau** for ~225 steps: all six games
  score the same, and the strategy is the equilibrium averaged over cards at
  each history (facing a bet: fold J/Q/K 0.51/0.52/0.49 vs 1.00/0.67/0.00).
  That's the cross-entropy optimum for a model that can't tell its card apart.
  Training KL sits at ~0.4.
- Seed 0 escaped at ~250: classic first (12.9% while the held-out games got
  *worse*), then everything by step 325. **Seed 1 never escaped** in 400 steps.

**Variant B** (kuhn3 + kuhn13, 400 steps): both seeds escape the plateau, at
step ~200-225 (earlier than A). Final: kuhn_lowball 2.0% / 2.3%, other held-out
games 1.6-4.8%, except kuhn_classic_234 at 8.5% / 11.1%. In kuhn13, 2/3/4 are the
weakest cards, and some of that label prior survives.

**Tests 1-3 on these adapters** (fixed letters; shuffled is the same):

| | best fit (wins of 7) | reverse list: sens / align | flip showdown: sens / align | rename shift | made-up labels, lowest wins |
|---|---|---|---|---|---|
| A distill (section 9) | label (4) | 0.67 / -0.02 | 0.06 / -0.12 | 0.24 | 42.6 / 54.2 |
| relabel A, seed 0 | **rules (7), TV 0.03** | 0.97 / 1.00 | 0.98 / 1.00 | 0.03 | **1.7 / 1.6** |
| relabel A, seed 1 | card-blind | 0.02 / -0.08 | 0.03 / -0.06 | 0.01 | 24.7 / 24.8 |
| relabel B, seed 0 | **rules (7), TV 0.07** | 1.02 / 0.99 | 1.09 / 0.98 | 0.06 | **2.7 / 1.3** |
| relabel B, seed 1 | **rules (7), TV 0.08** | 0.99 / 0.99 | 1.04 / 0.98 | 0.07 | **2.9 / 4.5** |

- The three adapters that escaped follow every line of the prompt by the right
  amount in the right direction (sensitivity ≈ 1, alignment ≈ 0.99), ignore
  renaming (shift ≈ the untrained model's 0.05), and play the made-up-label
  lowball probes at 1-5%: none of them seen in training.
- Seed 1 of A "fits" label recall only because it is card-blind everywhere,
  including on the reference games the label prediction comes from.
- So with randomized labels, the model reads the rules once it gets past the
  plateau: 3 of 4 runs did within 400 steps. Run ids: A `e73145ac`, `12e1ce9b`;
  B `d59b824d`, `66604751`.

## Takeaways so far

1. Without reasoning, both model sizes play the same exploitable "always
   aggressive" strategy (~16-17% on kuhn3).
2. Reasoning helps only with enough room to finish: harmful at 1.7b / 1024
   tokens, 7.0% at 4b / 6000 tokens. Cut-off traces are the main failure.
3. Qwen3's reasoning is long mostly because of hedging, not because it changes
   its mind; shorter reasoning is a cost goal more than an accuracy goal.
4. Imitating CFR reaches equilibrium on the training game and transfers
   negatively. Adding more fixed-label games doesn't fix this (54% → 40% on
   lowball at best).
5. Train against the full-vocabulary distribution, or the model learns to put
   probability on invalid answers.
6. Self-play cycles at every $\beta$ and window tried; the update rule, not
   the regularizer, needs fixing.
7. Trained models recall what card labels meant in training instead of reading
   the rank list and showdown rule (section 9). Randomizing labels in training
   fixes it: every run that got past an early card-blind plateau (3 of 4) reads
   the rules (tests 1-3) and plays every relabelled 3-card game, kuhn_lowball
   included, at 1-5%. Variant B (+ kuhn13) escaped sooner and in both seeds.

## Next

The plateau is accepted as a cost of label randomization (3 of 4 runs escape
within 400 steps; failed runs are easy to spot, since every game scores the
same). Next: strategic transfer. Every held-out game so far was kuhn3 under new
labels; the next test needs a game whose equilibrium really differs.
