# Pipeline: how the LLM plays, and how it is trained

The system has three layers: the **game** (ground truth), **how the LLM plays**
(evaluation, `mixstrat-eval`), and **how it is trained** (`mixstrat-train`).
Each step links to the code that does it. See `post_training_design.md` for the
reasoning behind the training plan and `experiment_log.md` for results.

---

## Part A: The game layer (ground truth, no LLM)

Everything the LLM is measured and trained against comes from here. It is exact
math, with no sampling.

**A1. Rules → `GameConfig`** ([config.py](../mixstrat/game/config.py))
A variant is a small dataclass: deck, antes, bet sizes, raise cap, who acts
first, high or low card wins, and who can see their own card. Presets live in
`configs/games.toml`. `deal_distribution()` gives $P(c_0, c_1)$, the probability
of each pair of cards being dealt.

**A2. Config → `GameTree`** ([tree.py](../mixstrat/game/tree.py))
- `_expand` builds every betting sequence. Each **node** is a betting history,
  with its legal actions (`check`/`bet` when not facing a bet,
  `fold`/`call`/`raise` when facing one) and its children.
- **Terminals** are folds or showdowns. `payoffs[k, c0, c1]` holds player 0's
  chips at terminal $k$ for every card pair at once; player 1 gets the negative.
- An **infoset** is what the acting player knows: their card plus the betting
  history, e.g. `P1 card=J | bet:0.5`. 3-card Kuhn has 12 infosets (6 per
  player); kuhn13 has 52.

**A3. Strategies → `Profile`** ([strategy.py](../mixstrat/game/strategy.py))
One `(cards × actions)` probability table per decision node: row $c$ is the
actor's action mix when holding card $c$. Every strategy, whether from CFR, the
LLM or uniform play, is a `Profile`, so everything downstream treats them
identically.

**A4. The reference answer: CFR+** ([cfr.py](../mixstrat/game/cfr.py))
Plays the game against itself (2000 iterations by default), tracking how much it
regrets not taking each action. Its *average* strategy converges to a Nash
equilibrium. Checked against the published Kuhn solution.

**A5. Scoring a strategy exactly**
- `expected_value` ([strategy.py](../mixstrat/game/strategy.py)): walks the
  tree, weighting every terminal by the deal probability and both players'
  action probabilities. The result is player 0's EV, the equivalent of
  infinitely many hands.
- `best_response` ([best_response.py](../mixstrat/game/best_response.py)): the
  perfect counter-strategy to one seat of a profile, and what it earns.
- `exploitability` ([best_response.py](../mixstrat/game/best_response.py)): the
  best responders' average gain over the game value, divided by the pot. It is
  0 only at equilibrium.
- `counterfactual_values` ([values.py](../mixstrat/game/values.py)): for every
  infoset $I$ and action $a$, the chips the player expects from taking $a$,
  given their card and that play reached this point:

  $$q(I,a) = \frac{\sum_o P(c,o)\,\pi_{\text{opp}}(n,o)\,V(n_a \mid c,o)}{\sum_o P(c,o)\,\pi_{\text{opp}}(n,o)}$$

  These are the rewards in stage 1b.

---

## Part B: How the LLM plays (`mixstrat-eval`)

**The key idea:** the model never plays hands. We read its *strategy* (its
action probabilities at every infoset), turn that into a `Profile`, and score it
exactly with Part A.

For each infoset in `tree.infosets` (`query_profile`,
[policy.py](../mixstrat/llm/policy.py)):

**B1. Build the prompt** (`build_prompt`, [prompts.py](../mixstrat/llm/prompts.py))
- **System message:** the rules, generated from the config by `rules_text`:
  deck, antes, bet sizes, who wins the showdown, and what each player can see.
- **User message:** "Your card: K.", the betting so far in chips ("Your
  opponent bet 1 chip."), the pot, and **only the legal actions**, lettered
  (A) check, B) bet 1 chip), then "Reply with a single letter."
- `Prompt.order` records which action each letter stands for. Letters can be
  shuffled to test for position bias.

**B2. Format it for the model**
- **Ollama:** the chat template is applied server-side. With thinking off,
  Qwen3's Ollama template appends ` /no_think` to the user message and leaves a
  blank line after the system header.
- **HF:** `render` ([hf.py](../mixstrat/llm/hf.py)) applies the official chat
  template, then reproduces Ollama's two differences so both backends see
  identical text.
- **Both:** the assistant turn is **prefilled** with
  `<think>\n\n</think>\n\nAnswer: **`, which closes the empty thinking block and
  leads straight into the answer letter.

**B3a. Direct mode: one forward pass**
1. Run the ~400-token prompt through the 28 layers. That gives a hidden vector
   at every position; only the last one is needed.
2. Output head: the last vector × the 151,936-row output matrix gives one logit
   per vocabulary token, computed in **float32** (`Float32Head`,
   [hf.py](../mixstrat/llm/hf.py)) because bf16 rounds logits to steps of
   0.125-0.25.
3. Softmax over the whole vocabulary.
4. Read the probabilities of the letter tokens only (`"A"`, `" A"`, `"B"`,
   `" B"`), add the spellings for each letter, and renormalize:
   `letter_logprobs_from_row` ([hf.py](../mixstrat/llm/hf.py)), or
   `letter_distribution` ([backend.py](../mixstrat/llm/backend.py)) for
   Ollama's top-20 list.
5. **Legal mass** is the total probability on the letters *before*
   renormalizing. It checks that the model is really choosing among the options.

**B3b. Reasoning mode: K samples per infoset** (`OllamaBackend.choose`,
[ollama.py](../mixstrat/llm/ollama.py); Ollama only)
1. `_reason`: generate with thinking on, under a "think briefly" instruction, a
   token budget and a fixed seed. Here the model *samples* text, and the trace
   is cut off if it hits the budget.
2. Feed the trace back inside `<think>…</think>`, followed by `Answer: **`,
   then do the B3a readout.
3. Repeat K times (4 so far) and average the letter distributions. Traces are
   cached and logged.

**B4. Letters → actions → `Profile`**
`to_action_order` ([policy.py](../mixstrat/llm/policy.py)) maps letter
probabilities back into tree action order using `prompt.order`. Doing this for
all infosets gives `Profile.from_infosets`: the LLM's complete strategy.

**B5. Score and report** ([evaluate.py](../mixstrat/llm/evaluate.py))
- Exploitability of the LLM, uniform and CFR.
- The LLM's EV in each seat against CFR.
- Per-infoset distance from the equilibrium (TV).
- Output: a terminal summary, plus with `--track` an MLflow run with live
  progress, `report.html` (the strategy matrices), `strategy.png`, the prompts
  and, for reasoning runs, `reasoning.html`.

---

## Part C: Post-training (`mixstrat-train`)

**C1. One-time setup** ([train/cli.py](../mixstrat/train/cli.py))
1. **Games:** the training game (kuhn_classic) and held-out games
   (kuhn_lowball), each solved with CFR+.
2. **Model:** Qwen3-1.7B in bf16 from Hugging Face, with the same prompts and
   readout as Part B.
3. **LoRA:** rank-16 adapters on every attention and MLP projection, 17.4M
   trainable parameters. The base weights are frozen.
4. **Memory:** gradient checkpointing (recompute activations in the backward
   pass instead of storing ~8 GB), micro-batches of 4 infosets, logits computed
   only at the last position. Peak is 5.2 GB.
5. **Optimizer:** AdamW, lr 1e-4 for distillation and 3e-5 for the value
   objectives (1e-4 saturates the policy in one Adam step).
6. **MLflow:** a run in the `post-training` experiment, with all settings as
   params.

**C2. Every training step: a batch of infosets, letters shuffled with a new seed**
Shuffling each step makes the model learn *actions*, not letter positions. With
one training game every infoset is used each step. With several (`--game a b c`),
`--batch-infosets N` samples N infosets per step, split evenly across the games
(`sample_items`, [train/common.py](../mixstrat/train/common.py)) so a 12-infoset
game isn't drowned out by 52-infoset ones.

**Stage 1a, `distill`** ([train/distill.py](../mixstrat/train/distill.py)):
imitate CFR.
1. Forward pass on the shuffled prompts → letter log-probabilities.
2. Loss per infoset:

   $$\underbrace{-\sum_a \sigma_{\text{CFR}}(a)\,\log p_\theta(a)}_{\text{match CFR}} \;+\; \underbrace{-\log(\text{legal mass})}_{\text{keep answers valid}}$$

3. Backward, then an AdamW step on the LoRA weights.

**Stage 1b, `value-vs-cfr` / `value-self-play`**
([train/value.py](../mixstrat/train/value.py)): learn from chip values.
1. **Snapshot the current strategy:** query the model on all infosets (no
   gradients, fixed letters) to get a `Profile`.
2. **Choose the world it is evaluated in** (`training_values`): the model plays
   its own seat, including its later decisions, and the opponent is **CFR**
   (1b-i) or **the model itself** (1b-ii).
3. **Exact action values:** $q(I,a)$ for every infoset from
   `counterfactual_values`, optionally averaged over the last N steps'
   opponents (`--window`). With `--value-refresh K` the snapshot and values are
   recomputed only every K steps, which matters for 13-card games, where one
   snapshot is ~100k tokens of forward passes.
4. **Forward pass** on the shuffled prompts → $p_\theta$.
5. **Loss per infoset** (each term fixes a failure we hit):

   $$\underbrace{-\sum_a \text{sg}\big(q_a - \bar q\big)\log p_\theta(a)}_{\text{NeuRD: push each action by its advantage}} + \underbrace{\beta\,\mathrm{KL}(p_\theta \Vert \text{uniform})}_{\text{magnet: stops cycling}} + \underbrace{-\log(\text{legal mass})}_{\text{valid answers}}$$

   - **NeuRD, not plain $-\sum p\,q$:** the plain version's gradient vanishes
     when $p_a \to 0$, so actions the model has stopped taking can't come back.
   - **Floor:** don't push an action down once $p_a < 10^{-3}$ (or up once
     $> 1-10^{-3}$). Otherwise NeuRD drives losing actions toward zero
     probability forever and starves the spots that need to change.
   - **Magnet:** the stabilizer from magnetic mirror descent. A uniform magnet
     is an entropy bonus, which keeps mixed strategies alive.
6. Backward, then an AdamW step.

**C3. Every 10 steps: evaluate**
Run Part B (B1-B5) with the HF backend in eval mode on the training game *and*
the held-out games. Log exploitability, seat EVs, mean TV and legal mass, and
warn if legal mass drops below 0.95.

**C4. At the end**
Save the LoRA adapter to `outputs/checkpoints/<run>/` (score it any time with
`mixstrat-eval --backend hf --adapter …`). Log the full report artifacts for
each game under `final/<game>/` in the MLflow run.

---

**C5. Sweeps** ([train/sweep.py](../mixstrat/train/sweep.py))
`mixstrat-sweep <name>` runs every combination defined in
[configs/sweeps.toml](../configs/sweeps.toml), one `mixstrat-train` at a time.
Runs are tagged `sweep=<name>` in MLflow, logged to
`outputs/logs/sweeps/<name>/`, and skipped on a rerun once finished, so an
interrupted sweep resumes with the same command. `--dry-run` lists the runs and
the estimated time.

## How the pieces connect

```
GameConfig ─▶ GameTree ─┬─▶ CFR+ ─▶ equilibrium Profile ──────────────┐
                        │                                             │
                        ├─▶ build_prompt(infoset) ─▶ LLM ─▶ letter    │
                        │      (rules, card,       (prefill,  probs   │
                        │       history, A/B/C)     fp32 head)  │     │
                        │                                       ▼     ▼
                        │                        LLM Profile ─▶ exploitability,
                        │                                       EV vs CFR, reports
                        │
                        └─▶ training: loss(letter probs, CFR target or q-values)
                                       ─▶ backprop ─▶ LoRA weights ─▶ (repeat)
```

## Not built yet

- **Reasoning-mode training (stage 2).** All training so far is direct mode:
  one forward pass, no reasoning. Training on reasoning traces needs text
  generation inside the training loop (GRPO).
- **Stable self-play.** With the magnet at β = 0.02, self-play cycles between
  pure strategies.
