# Art-20Q Experiment Runner Notebook

## Run 1 — 2026-02-24

**Hypothesis:** GRPO from the base Qwen2.5-14B-Instruct model will exhibit policy collapse within 50 training steps. The base model already has basic tool-use ability from pretraining; GRPO may briefly improve accuracy before collapsing to a degenerate policy. This run establishes the baseline failure pattern.

**Setup:**
- Model: Qwen/Qwen2.5-14B-Instruct (no SFT warm-start)
- RL: GRPO via ART ServerlessBackend
- Reward: v5 (narrowing bonus, suicide-guess penalty)
- Prompt: v4 (attribute hallucination guard)
- Batch size: 6 secrets × 10 rollouts/secret = 60 trajectories/step
- Learning rate: 1e-5
- Steps: 50 (seed 1 only; seeds 2-5 cancelled after observing collapse)
- Eval: 20 episodes at steps 0, 10, 20 (+ final)

**Results:**

| Metric | Step 0 (baseline) | Step 10 (peak) | Step 20 (collapsed) | Step 30 (terminal) |
|--------|-------------------|----------------|---------------------|---------------------|
| Accuracy | 30% (6/20) | 65% (13/20) | 0% (0/20) | 0% (0/20) |
| Avg questions | 12.7 | 10.5 | 0.2 | 0.0 |
| Avg candidates remaining | 2.6 | 1.8 | 74.4 | 76.0 |
| Avg reward | -1.4 | +10.3 | 0.0 | 0.0 |
| Guessed | 14/20 | 17/20 | 0/20 | 0/20 |
| Timeout | 6/20 | 3/20 | 20/20 | 20/20 |

Collapse timeline from W&B train metrics:
- Steps 1-5: Reward noisy, questions ~10, learning begins
- Steps 5-10: Peak performance. Reward reaches +6, correct rate ~0.5, questions ~10-12
- Steps 10-15: Questions start dropping. Reward declining. Entropy falling.
- Steps 15-20: Full collapse. Questions → 0, reward → 0, reward_std_dev → 0
- Steps 20+: Degenerate loop. All trajectories identical. No gradient signal.

Failure mode: "Safe haven" timeout. Agent loops `list_attributes → get_candidate_count` indefinitely, never calling `ask_yesno` or `submit_guess`. Reward 0.0 from timeout is higher than negative rewards from wrong guesses, so GRPO reinforces inaction.

Cost: ~$25 (seed 1 only, 25 steps before kill + remaining steps)

**Skill acquisition:**

| Skill | Step 0 | Step 10 | Step 20 | Step 30 |
|-------|--------|---------|---------|---------|
| 1. Tool use (calls list_attributes) | 20/20 | 20/20 | 20/20 | 20/20 |
| 2. Sequencing (asks before guessing) | 20/20 | 20/20 | 2/20 | 0/20 |
| 3. State awareness (checks candidates) | 14/20 | 19/20 | 20/20 | 20/20 |
| 4. History tracking (no redundant attrs) | 14/20 | 16/20 | 20/20 (vacuous) | 20/20 (vacuous) |
| 5. Entropy maximization | not measured | not measured | N/A | N/A |

Key observations:
- Skills 1-2 came from pretraining (present at step 0). GRPO did not need to teach them.
- Skill 3 improved under GRPO (14→19/20 by step 10). This is a genuine RL acquisition.
- At collapse (step 20), skill 3 reads 20/20 but is vacuous — the agent calls `get_candidate_count` obsessively without acting on the information. Skill 2 (sequencing) is destroyed.
- The collapsed agent's behavior is pathological: `list_attributes, get_candidate_count, list_attributes, get_candidate_count, ...` repeated 12+ times per episode. It "uses tools" and "checks state" but does nothing with either.

**Interpretation:**

This confirms the thesis premise: GRPO from a cold start cannot sustain multi-step agent behavior on this task.

The step-10 peak is the most interesting data point. At 65% accuracy, the base model + 10 GRPO steps **matched gpt-4o** (63.3%). This happened because:
1. The base Qwen-14B already understood tool use and question-asking from pretraining (skills 1-2 at step 0)
2. GRPO sharpened state awareness (skill 3) and reduced average questions from 12.7→10.5
3. The model was already "close" — GRPO refined execution, it didn't teach reasoning

The collapse is equally informative. GRPO's trajectory-level reward cannot distinguish between "good timeout" (asked 15 careful questions, still couldn't narrow) and "bad timeout" (did nothing). Both get reward ≈ 0. Once a few trajectories discover the safe-haven loop, the low variance makes it an attractor.

The reward_std_dev chart tells the whole story: it drops from ~12 to ~0 as all trajectories converge to the same degenerate behavior. At std_dev ≈ 0, GRPO's advantage computation produces no gradient. The policy is frozen in the collapsed state with no mechanism to escape.

This is NOT a hyperparameter issue. It's a structural property of trajectory-level reward + on-policy RL: degenerate low-variance policies are stable equilibria.

**Next:**

Run 2 — SFT warm-start + GRPO. The hypothesis: if SFT pre-trains the model to reliably ask questions and guess (making skill 2 "sticky"), GRPO's optimization landscape changes. The safe-haven loop becomes a local minimum the model must actively degrade toward, rather than a default it falls into. The SFT checkpoint starts at oracle-level play; GRPO's job is to maintain or refine, not discover. Target: 70-85% accuracy sustained through 50 steps without collapse.

---

## Run 2a (failed) — 2026-02-24

**Hypothesis:** SFT on oracle trajectories teaches the model to play 20Q optimally (question-asking strategy + correct guessing). Post-SFT accuracy should be high before GRPO begins.

**Setup:**
- Model: `run2-sft` based on OpenPipe/Qwen3-14B-Instruct
- SFT: 76 oracle trajectories × 3 epochs = 228 examples, batch_size 2, peak_lr 2e-4, cosine schedule
- Training via `art.utils.sft.train_sft_from_file` (ART 0.5.11)

**Results:**

SFT training metrics looked healthy:
- Loss: 0.45 → 0.03 (steady decline over 114 gradient steps)
- grad_norm: nonzero throughout (0.1-0.6 range)
- Model step advanced: 0 → 1
- ~400-500 trainable tokens per batch (assistant tokens being trained on)

Post-SFT eval (20 episodes):
- Accuracy: **0% (0/20)**
- Wrong guesses: 20/20, Timeouts: 0/20
- Avg questions: 7.0 (oracle is 6.6 — nearly identical)
- Avg candidates remaining: 1.1 (near-perfect narrowing)

**Failure mode: Object ID hallucination.**

The model learned the oracle's question-asking strategy almost perfectly — it asks ~7 information-theoretic questions and narrows to ~1 candidate. But when it calls `submit_guess`, it generates **fabricated object IDs** like `n1k3l`, `u8d0e`, `c4v6w` that don't exist in the dataset. Real IDs are arbitrary 5-character strings (e.g., `a1x9z` for Lion), so the model learned the format but not the mapping.

**Root cause:** The oracle has direct access to `ep["candidates"]` and never needs to call `get_top_candidates` to see remaining IDs. The SFT data therefore contained no `get_top_candidates` calls before `submit_guess`. The model had no way to learn that it should look up the actual candidate IDs before guessing — it just hallucinated plausible-looking ones.

**Fix:** Modified `generate_sft_data.py` to inject a `get_top_candidates(k=N)` call immediately before every `submit_guess`. This teaches the model the pattern: narrow down → look up remaining IDs → guess from the list.

**Additional issue:** W&B logging went to project `20q` instead of `art-20q-runner-2026`. Fixed in `train_sft.py`.

**Interpretation:**

This is not an RL finding — it's a data engineering lesson about the oracle-to-LLM gap. The oracle operates with perfect state access; the LLM operates through tool outputs. SFT data must reflect the LLM's information constraints, not the oracle's. Any action the oracle takes using internal state that the LLM can only access via tools needs an explicit tool call injected into the training data.

This is actually relevant to the broader thesis: even with SFT, the training data needs to account for the difference between "having knowledge" and "having access to knowledge through an interface." The oracle knows the answer; the LLM must look it up.

**Cost:** ~$5 (SFT training only, GRPO not started)

**Next:** Re-run SFT as `run2-sft-v2` with fixed oracle data that includes `get_top_candidates` before `submit_guess`.

---

## Run 2b (SFT only, pre-GRPO) — 2026-02-24

**Hypothesis:** SFT on fixed oracle trajectories (with `get_top_candidates` before `submit_guess`) will produce a model that plays near-optimally before any GRPO training.

**Setup:**
- Model: `run2-sft-v2` based on OpenPipe/Qwen3-14B-Instruct
- SFT: 76 oracle trajectories × 3 epochs = 228 examples
- Batch size: 2, peak_lr: 2e-4, cosine schedule, 114 gradient steps
- Fix from 2a: `get_top_candidates` injected before every `submit_guess`

**Results:**

Post-SFT eval (20 episodes):
- **Accuracy: 95% (19/20)**
- Wrong: 0, Timeout: 1
- Avg questions: 7.7 (oracle: 6.6)
- Avg candidates remaining: 1.1

SFT training metrics:
- Loss: 0.45 → 0.03 (healthy convergence)
- grad_norm: nonzero throughout
- Model step: 0 → 1

**Skill acquisition:**

The SFT model exhibits all 5 skills:
1. Tool use: 20/20 (calls list_attributes)
2. Sequencing: 19/20 (asks before guessing, 1 timeout)
3. State awareness: 20/20 (calls get_top_candidates before guessing)
4. History tracking: presumed high (7.7 avg questions without redundancy at this accuracy)
5. Entropy maximization: partially — 7.7 avg questions vs oracle's 6.6 suggests near-optimal but not perfect attribute selection

**Interpretation:**

This is a landmark result for the thesis. 76 SFT trajectories on a 14B model achieved **95% accuracy** — matching gpt-5.2 (96.7%) and far exceeding gpt-4o (63.3%). Key implications:

1. **SFT trivially teaches what GRPO struggled to learn.** Run 1 GRPO briefly hit 65% at step 10 before collapsing. SFT hit 95% in a single training pass. The skills GRPO needed to discover through reward signal (sequencing, state awareness, strategic questioning) were directly imprinted by supervised learning.

2. **The base model already has the reasoning capacity.** Qwen-14B can execute entropy-maximization strategy when shown how — it doesn't need to discover it. This supports the thesis: "the base model needs to already understand the strategy — RL refines execution, it doesn't teach reasoning."

3. **The remaining question is whether GRPO preserves or degrades this.** 95% is the ceiling. If GRPO from this checkpoint maintains ≥90%, SFT+GRPO is a viable pipeline. If it collapses again, that tells us GRPO is actively destructive even from a strong starting point.

**Cost:** ~$5 (SFT training) + ~$2 (eval)

**Next:** Run GRPO from the `run2-sft-v2` checkpoint (50 steps, single seed). The critical metric is whether accuracy stays above 85% or collapses as in Run 1.

---

## Run 3 — Free-Form Questions — 2026-02-24

**Why pivot to Run 3 now (before Run 2c GRPO)?**

Run 2b's SFT result (95% accuracy) effectively answers the Run 2 question: SFT trivially teaches the predefined-attribute strategy. The remaining Run 2c step (GRPO from SFT checkpoint) tests whether GRPO preserves that — an important but incremental question. Run 3 asks something more fundamental.

With predefined attributes, 20Q is a solved algorithm problem. The agent picks from a menu of 59 attributes; optimal play is a greedy entropy search. SFT can teach this perfectly because the action space is closed. But real agent tasks don't have menus — agents must generate actions in natural language. Run 3 removes the attribute menu entirely, replacing `ask_yesno(attr_name)` with `ask_question(question)` where the agent writes any yes/no question it wants.

This transforms 20Q from an algorithm problem into an agent problem:
- Action space: open-ended (natural language) instead of pick-from-list
- Strategy: requires world knowledge ("Is it alive?" requires knowing what objects are alive) + information theory (which questions bisect the candidate set)
- No oracle: there's no clean way to generate gold free-form trajectories, so SFT warm-start isn't available as an escape hatch

The thesis question becomes sharper: **Can GRPO teach an agent to ask good questions, or only to follow a script?**

---

### Run 3a (failed — infrastructure) — 2026-02-24

**Hypothesis:** Same as Run 3 overall — GRPO from base Qwen-14B cannot learn discriminating freeform questions from reward signal alone.

**Setup:**
- Model: OpenPipe/Qwen3-14B-Instruct (base, no SFT)
- RL: GRPO via ART ServerlessBackend
- Reward: v5 (same as Run 1 — narrowing bonus, suicide-guess penalty)
- Prompt: v6 (freeform — no attribute list, no `list_attributes()`)
- Tools: `ask_question(question)`, `get_candidate_count()`, `get_top_candidates(k)`, `submit_guess(object_id)`
- LLM judge: gpt-5-nano evaluates each question against each candidate object
- Concurrency: `asyncio.Semaphore(20)` — max 20 concurrent judge calls
- Batch size: 6 secrets × 10 rollouts/secret = 60 trajectories/step
- Learning rate: 1e-5
- Steps: 5 (smoke test), seed 1
- W&B run: `run3-smoke-seed1`

**Results:** Killed before step 1 completed. A single GRPO step could not finish in reasonable time.

**Failure mode: Infrastructure bottleneck, not RL failure.**

Each freeform question requires 77 LLM judge calls (1 for the secret + 76 for all candidates), versus zero external calls in predefined mode. With batch size 6, 10 rollouts per secret, and ~8 questions per episode, a single GRPO step requires ~37,000 judge calls. At semaphore 20, this meant ~4 serial batches per question, making each step take prohibitively long.

This is a new class of failure mode not seen in Runs 1-2: the environment itself becomes the bottleneck. In predefined mode, candidate filtering is a fast in-memory operation (loop over 76 objects, check a boolean). In freeform mode, it's 76 API calls per question. The 20Q environment went from O(1) per question to O(N × API_latency) per question.

**Cost:** Minimal (killed early, some gpt-5-nano calls wasted).

**Interpretation:** Not an RL finding — purely an infrastructure lesson. Freeform environments with LLM judges have fundamentally different cost and latency profiles from structured environments. Any freeform agent task that requires per-candidate evaluation will hit this wall. The fix is caching + higher concurrency, not changes to the RL algorithm.

**Next:** Fix infrastructure: add judge result cache, bump concurrency, retry smoke test.

---

### Run 3b (post-cache fix) — 2026-02-24

**Hypothesis:** GRPO from the base Qwen-14B model cannot learn to generate discriminating natural language yes/no questions from reward signal alone. Without the predefined attribute menu, the agent must compose its own questions — a creative generation task that requires both world knowledge and information-theoretic reasoning. We expect accuracy well below the predefined-mode baseline (Run 1 step-0: 30%), with the agent producing vague or repetitive questions that fail to narrow candidates efficiently.

**Setup:**
- Model: OpenPipe/Qwen3-14B-Instruct (base, no SFT)
- RL: GRPO via ART ServerlessBackend
- Reward: v5 (same as Run 1 — narrowing bonus, suicide-guess penalty)
- Prompt: v6 (freeform — no attribute list, no `list_attributes()`)
- Tools: `ask_question(question)`, `get_candidate_count()`, `get_top_candidates(k)`, `submit_guess(object_id)`
- LLM judge: gpt-5-nano evaluates each question against each candidate object
- Concurrency: `asyncio.Semaphore(50)` — bumped from 20
- Judge cache: in-memory `_judge_cache` keyed by `(object_id, normalized_question)` — eliminates redundant calls across GRPO rollouts
- Batch size: 6 secrets × 10 rollouts/secret = 60 trajectories/step
- Learning rate: 1e-5
- Steps: 5 (smoke test), seed 1

**Changes from Run 3a:**
1. **Concurrency semaphore: 20 → 50.** Reduces serial batches per question from ~4 to ~2.
2. **Judge result cache.** `_judge_cache` keyed by `(object_id, question.strip().lower())`. GRPO generates 10 rollouts per secret — similar questions across rollouts (e.g., "Is it alive?") now hit cache. This is the biggest efficiency win since many rollouts converge on the same popular questions.
3. **`evaluate_question` signature change.** Added `object_id` parameter to support cache keying. `ask_freeform` updated to pass object IDs through.
4. **Safety net for judge inconsistency.** If the judge gives different answers for the same question on the secret vs. during candidate filtering, the secret is force-added back to the candidate list. Prevents the agent from being punished for judge noise.

**Key design decisions (unchanged from 3a):**
1. LLM judge sees the object's name + all 59 boolean attributes + the question → returns yes/no/unknown. This allows questions that span multiple attributes ("Is it something you'd find in a kitchen?").
2. Candidate filtering: judge evaluates every remaining candidate per question, keeps those matching the secret's answer. Parallel via `asyncio.gather()`.
3. "unknown" answers penalized at 0.5 (same as invalid attributes in v5), incentivizing the agent to ask clear, answerable questions.
4. No SFT warm-start. Starting from base to isolate what GRPO alone can learn in freeform mode.

**Cost note:** Freeform mode has a fundamentally different cost profile from predefined mode. Each GRPO step costs gpt-5-nano judge calls proportional to `batch_size × rollouts × questions_per_episode × num_objects`. At ~37K calls/step (pre-cache) and 50 steps, budget monitoring is critical. The judge cache should reduce actual API calls significantly for repeated questions.

**Expected results:**
- Step 0 (base model): ~10-20% accuracy. The base model can ask questions but won't have a strategic narrowing approach.
- Steps 1-50: Unclear. If GRPO can learn freeform questioning, we should see candidate reduction improve. If it can't, expect collapse to vague questions or immediate guessing.
- Key metric to watch: average candidate count after questions. If the judge is working and questions are discriminating, this should decrease over training.

**What this tells us about the thesis:**
- If GRPO learns to ask good freeform questions → the thesis needs refinement. GRPO can acquire creative generation strategies, not just execute scripted ones.
- If GRPO fails (collapse or no improvement) → strong evidence that open-ended action spaces require SFT demonstration, supporting the thesis that "RL refines execution, it doesn't teach reasoning."
- If base model is already decent at step 0 → pretraining knowledge matters more than RL, same as Run 1's step-0 showing.

**Results (smoke test: `run3-smoke-v3-seed1`, 20 objects, 5 steps):**

| Metric | Step 0 (baseline) | Step 5 (mid eval) | Final eval (50 eps) |
|--------|-------------------|-------------------|---------------------|
| Accuracy | 25.0% (5/20) | 25.0% (5/20) | 20.0% (4/20) |
| Wrong guesses | 15 | 15 | 16 |
| Timeouts | 0 | 0 | 0 |
| Avg questions | 6.5 | 6.0 | 5.6 |
| Avg candidates remaining | 1.0 | 1.0 | 1.1 |

All 5 training steps completed (100% success rate, 0 failures). Infrastructure fixes worked — the run finished end-to-end with 20 objects.

**Observations:**

1. **No collapse.** Unlike Run 1 (which collapsed to inaction by step 15), the freeform agent keeps asking questions and guessing through all 5 steps. Zero timeouts. The v5 reward function's timeout penalty (-5) vs wrong guess penalty (-15) doesn't create the safe-haven attractor here — possibly because the freeform prompt (v6) starts with "ask questions" rather than "call list_attributes()", giving the base model a more natural entry point.

2. **Questions are discriminating.** The agent asks ~6 questions and narrows from 20 candidates to ~1. This is near-optimal for 20 objects (log2(20) ≈ 4.3). The base Qwen-14B model already knows how to ask good binary-search-style questions ("Is it alive?", "Is it man-made?") from pretraining. This mirrors Run 1's finding: skill 1-2 (tool use, sequencing) come from pretraining, not RL.

3. **Accuracy is only 20-25% despite narrowing to 1 candidate.** This is the critical anomaly. If the agent narrows from 20 to 1 candidate and guesses that candidate, accuracy should be near 100%. The fact that it's 25% means the judge is introducing systematic noise during candidate filtering, eliminating the secret object in ~75% of games despite the safety net.

4. **No GRPO signal.** Accuracy flat at 25% across 5 steps. Too few steps to be conclusive, but the reward distribution during training showed mostly negative rewards (wrong guesses), meaning the GRPO advantage computation has limited positive signal to amplify.

**The LLM judge problem — a deeper look:**

The core issue is how freeform questions get evaluated. Here's the exact flow:

When the agent calls `ask_question("Is it alive?")`, the environment does the following:

```python
# Step 1: Evaluate the question against the SECRET object
secret = objects_by_id[ep["secret_id"]]
secret_answer = await evaluate_question(
    ep["secret_id"], secret["name"], secret["attrs"], question
)
# e.g., secret is "Dog", judge returns "yes"

# Step 2: Evaluate the SAME question against ALL remaining candidates
results = await asyncio.gather(
    *(evaluate_question(oid, obj["name"], obj["attrs"], question)
      for oid in ep["candidates"])
)

# Step 3: Keep only candidates whose answer MATCHES the secret's answer
ep["candidates"] = [oid for oid, answer in results if answer == secret_answer]
# e.g., keep all objects where judge also said "yes" to "Is it alive?"
```

Each `evaluate_question` call sends this prompt to gpt-5-nano:

```
You are a yes/no question judge for a 20 Questions game.

Object: {object_name}
Attributes (true means the object has this property):
  is_animal: True
  is_large: True
  has_legs: True
  ...

Player's question: "Is it alive?"

Based on the object's attributes and your general knowledge about
"{object_name}", answer: yes, no, or unknown.
```

The problem is **judge inconsistency across objects**. For a question like "Is it bigger than a car?", gpt-5-nano might:
- Say "yes" for "Elephant" (correct)
- Say "yes" for "Dog" (wrong — dogs are smaller than cars)
- Say "no" for "Whale" (wrong — whales are bigger than cars)

If the secret is "Elephant" and the judge says "yes", then any candidate where the judge incorrectly said "no" gets eliminated — even if it should have been kept. Conversely, candidates that should be eliminated might survive if the judge incorrectly agrees.

The safety net catches one case — if the secret itself gets filtered out:

```python
# Safety: ensure the secret is always in candidates
if ep["secret_id"] not in ep["candidates"]:
    ep["candidates"].append(ep["secret_id"])
```

But it can't catch the general case where noise in candidate filtering leads to the wrong final candidate. After 6 questions, each with independent judge noise across 20 objects, the cumulative error is enough to make the surviving candidate wrong ~75% of the time.

This is a **fundamental limitation of the LLM-judge approach**: the judge must be consistent across all 76 (or 20) objects for the same question. A small per-object error rate compounds multiplicatively across questions. With 6 questions and 20 objects, even a 5% per-call error rate means ~50% of games have at least one filtering error.

**Comparison to predefined mode:** In predefined mode, `ask_yesno("is_animal")` does a deterministic boolean lookup — zero noise, zero cost, instant. The freeform mode pays a massive tax in cost, latency, AND accuracy for the privilege of open-ended questions.

**Implication for the thesis:** The judge noise means we can't cleanly separate "GRPO failing to learn strategy" from "GRPO succeeding at strategy but being undermined by a noisy environment." The 25% accuracy ceiling may be an environment problem, not an RL problem. To isolate the RL question, we'd need either (a) a perfect judge, or (b) a way to measure question quality independently of final accuracy.

**Next:**
- Option A: Launch full 50-step run with 20 objects to see if GRPO can improve even within the noisy ceiling. If accuracy moves from 25% → 40%, that's signal even if the ceiling is low.
- Option B: Improve the judge first. Options: use a stronger model (gpt-5-mini), add self-consistency (majority vote of 3 judge calls), or pre-compute a ground-truth answer matrix for common questions.
- Option C: Measure question quality directly — track average candidate reduction per question across training steps. If GRPO learns to ask better questions, candidate reduction should improve even if final accuracy doesn't (due to judge noise).

---

### Run 3c (full object set, optimized judge) — 2026-02-24

**Hypothesis:** Run 3b showed the base model already asks near-optimal questions on 20 objects, but judge noise capped accuracy at ~25%. Run 3c scales to the full 76 objects to test whether (a) the base model's question quality holds at full scale, and (b) GRPO produces any training signal when the base model is already competent. We expect candidate reduction per question to be flat across training steps (no GRPO learning), and accuracy to remain bounded by judge noise.

**Setup:**
- Model: OpenPipe/Qwen3-14B-Instruct (base, no SFT)
- RL: GRPO via ART ServerlessBackend
- Reward: v5 (narrowing bonus, suicide-guess penalty)
- Prompt: v6 (freeform)
- Steps: 5 (smoke test), seed 1
- **All 76 objects** (vs 20 in Run 3b)
- W&B run: `run3-smoke-v4-seed1`

**Changes from Run 3b:**
1. **Full 76-object set** — tests whether question quality degrades with more candidates and whether judge noise compounds worse at scale (76 judge calls/question vs 20).
2. **Judge cache pre-warming** — 20 common questions × 76 objects = 1,520 entries cached before training. Eliminates cold-start latency for frequent questions like "Is it alive?".
3. **Trimmed judge prompt** — reduced token count for faster inference.
4. **Higher concurrency** — semaphore bumped to 100 for parallel judge calls.

**Results (smoke test: `run3-smoke-v4-seed1`, 76 objects, 5 steps):**

| Metric | Step 0 (baseline) | Step 1 | Step 2 | Step 3 | Step 4 | Step 5 |
|--------|-------------------|--------|--------|--------|--------|--------|
| Accuracy | 30.0% (6/20) | — | — | — | — | TBD |
| Avg questions | 9.7 | 9.5 | 9.4 | 9.2 | 10.1 | TBD |
| Avg final candidates | 5.9 | 3.5 | 9.5 | 5.8 | 5.1 | TBD |
| Avg reward | — | -10.8 | -0.6 | -1.9 | -4.0 | TBD |
| Correct rate (train) | — | 0.15 | 0.42 | 0.37 | 0.33 | TBD |

_Step 5 in progress (82% gathered). Mid-training and final eval TBD._

**Preliminary observations:**

1. **Baseline accuracy 30% on 76 objects** (vs 25% on 20 objects in Run 3b). Surprisingly not worse at full scale — the base model handles 76 objects as well as 20.

2. **Higher avg candidates remaining (5.9 vs 1.0 in Run 3b).** With 76 objects, the agent needs more questions to narrow down, and judge noise has more room to compound. 5.9 remaining after ~10 questions means the agent is still narrowing significantly (76→6 is ~3.7 halvings) but not reaching the single-candidate precision seen with 20 objects.

3. **Training metrics show high variance.** Reward swings from -10.8 (step 1) to -0.6 (step 2) to -4.0 (step 4). Correct rate fluctuates 0.15-0.42. This noise is consistent with judge inconsistency dominating the reward signal.

4. **No clear GRPO learning trend yet.** Questions asked stays ~9-10, candidate reduction shows no monotonic improvement. Need final eval to compare step-0 vs step-5 properly.

**Skill acquisition:** TBD (pending final eval trajectories)

**Interpretation:** TBD (pending completion)

**Cost:** TBD

---

### Run 3 — Post-Mortem (shelved) — 2026-02-24

**Decision:** Shelving Run 3 (free-form questions). The approach has two compounding problems that make it unsuitable for testing the thesis.

**Problem 1: Low throughput makes GRPO training impractical.**

Each freeform question requires N judge calls (one per remaining candidate). With 76 objects, that's ~77 calls for the first question, ~38 for the second, etc. — roughly 150 judge calls per episode. A single GRPO step (6 secrets × 10 rollouts = 60 episodes) needs ~9,000 judge calls. Even with all optimizations applied (cache pre-warming of 1,520 entries, semaphore at 100, trimmed prompts), a 5-step smoke test could not complete its final evaluation within reasonable time. The run hung on step 5's eval phase indefinitely.

For comparison: predefined mode runs a full 50-step training in ~2 hours. Freeform mode couldn't finish 5 steps. The 50-step run needed for meaningful GRPO signal would take days and cost significantly more in both judge API fees and ART compute time (the model sits idle while waiting for judge calls).

The infrastructure optimizations helped but couldn't overcome the fundamental O(N × API_latency) per question:

| Optimization | Impact |
|---|---|
| Judge cache (cross-rollout) | ~5-8x reduction for repeated questions |
| Pre-warm 20 questions × 76 objects | Eliminated cold-start for common questions |
| Semaphore 20→100 | ~2x throughput improvement |
| Trimmed prompt (true attrs only) | ~60% fewer input tokens, marginal speed gain |
| **Net effect** | Still too slow for 50-step training |

**Problem 2: Unreliable judge makes reward signal uninterpretable.**

Evidence from trajectory sampling (Run 3b, 5 eval episodes from base model):

| Episode | Object | Questions | Final candidates | Correct | Behavior |
|---|---|---|---|---|---|
| 1 | Ice Cream | 8 | 1 | Yes | Clean narrowing: "Is it a living thing?" → "Is it man-made?" → "Is it something you can eat?" → correct |
| 2 | Microwave | 0 | 76 | No | `get_candidate_count` loop ×25, never asked a question |
| 3 | Paper | 0 | 76 | No | `get_candidate_count` loop ×25, never asked a question |
| 4 | Hat | 11 | 1 | Yes | Good narrowing but asked "Is it something you wear on your head?" twice (redundant) |
| 5 | T-shirt | 0 | 76 | No | `get_candidate_count` loop ×25, never asked a question |

The 2 successful episodes show the base model already asks good questions from pretraining — "Is it a living thing?", "Is it man-made?", "Is it something you can eat?" are textbook binary search. But 3/5 episodes degenerate to the `get_candidate_count` safe-haven loop (same failure as Run 1).

More critically, judge accuracy testing revealed systematic failures:

| Question | Object | Expected | Judge said |
|---|---|---|---|
| "Is it an animal?" | Dog | yes | **yes** |
| "Is it an animal?" | Car | no | **no** |
| "Is it man-made?" | Dog | no | **unknown** |
| "Is it man-made?" | Car | yes | **unknown** |

The judge returned "unknown" for basic questions like "Is it man-made?" because gpt-5-nano's reasoning tokens couldn't reliably infer the answer from the attribute list (no `is_man_made` attribute exists — the judge must use general knowledge). This means:
- Questions that should bisect the candidate set instead get penalized as invalid (-0.5)
- Candidate filtering is noisy: the wrong objects survive or get eliminated
- The agent's reward signal reflects judge quality, not question quality

Run 3b confirmed this at scale: accuracy was 25% despite narrowing to ~1 candidate. The agent asked good questions and narrowed correctly, but judge noise during filtering led to the wrong final candidate ~75% of the time.

**Why this matters for the thesis:**

The thesis question for Run 3 was: "Can GRPO learn to ask discriminating free-form questions?" We can't answer it because:

1. The reward signal is dominated by judge noise, not agent quality. GRPO would be optimizing for "ask questions the judge happens to evaluate consistently" rather than "ask questions that maximally split candidates."
2. We can't distinguish "GRPO failed to learn" from "GRPO learned but the judge undermined it."
3. The base model already asks good questions (episodes 1 and 4 show near-optimal binary search from pretraining), so there's limited room for GRPO to improve even with a perfect judge.

**What we learned anyway:**

1. **LLM-as-judge environments have fundamentally different economics.** Predefined mode: O(1) per question, deterministic. Freeform mode: O(N × API_latency) per question, stochastic. This isn't a hyperparameter issue — it's a structural incompatibility with GRPO's need for high-volume rollouts.

2. **Judge consistency matters more than judge accuracy.** A judge that's wrong but consistent across objects still produces valid candidate filtering. A judge that's 95% accurate but inconsistent across objects introduces multiplicative noise that compounds across questions. For 20Q filtering, consistency > accuracy.

3. **The base model already has freeform questioning from pretraining.** Episodes 1 and 4 show Qwen-14B asking "Is it a living thing?", "Is it man-made?", "Is it something you can eat?" without any training. This mirrors Run 1's finding: skills 1-3 come from pretraining, not RL.

4. **Open-ended action spaces don't automatically make problems harder for the agent.** The agent's question quality was fine — the environment's evaluation was the bottleneck. This is an important design lesson: freeform environments need reliable judges before they can test RL capabilities.

**What would make Run 3 viable:**

- A deterministic judge (pre-computed answer matrix for all object × question pairs) — but this collapses back to predefined mode
- A much stronger judge model (gpt-5.2) — but at $1.75/1M input tokens, the cost for 50 GRPO steps would exceed the entire project budget
- Self-consistency voting (3 judge calls, majority vote) — reduces noise but triples cost and latency
- Smaller object set (8-10 objects) — reduces judge calls but also reduces the RL challenge

None of these solve both problems simultaneously within budget.

---

## Run 4a — Answer Corruption Smoke Test — 2026-02-24

**Hypothesis:** The SFT-trained model (Run 2, ~95% on clean episodes) will degrade under 15% answer corruption. GRPO may teach the model to detect inconsistencies (e.g., candidate count not decreasing as expected) and adapt — using `get_candidate_count` to verify, re-asking attributes, or guessing earlier when uncertain. If SFT+GRPO holds higher accuracy than SFT-only under corruption, GRPO taught resilience.

**Setup:**
- Model: `run2-sft-v2` checkpoint (SFT-trained, ~95% clean accuracy)
- RL: GRPO via ART ServerlessBackend
- Reward: v5
- Prompt: v4
- Perturbation: answer_corruption at 15% rate
- Steps: 5 (smoke test), seed 42
- 20 objects (subset)
- W&B run: `run4a-smoke-seed42`

**Results:**

| Metric | Step 0 (SFT baseline) | Step 5 (SFT+GRPO) |
|--------|----------------------|-------------------|
| Accuracy | 10% (2/20) | 5% (1/20) |
| Wrong guesses | 16 | 14 |
| Timeouts | 2 | 5 |
| Avg questions | 6.3 | 7.3 |
| Avg candidates remaining | 0.8 | 0.7 |
| Avg corrupted questions | ~0.95/episode | ~1.17/episode |

**W&B Training Charts (5 steps):**

- **train/correct:** Oscillates 0.0–0.1 across all steps. Nearly zero positive signal for GRPO to amplify. The one spike to ~0.1 at step 4 doesn't sustain.
- **train/reward:** Stuck at -8 to -9.5, all negative, no upward trend. The model is consistently penalized.
- **train/reward_std_dev:** Starts ~5, dips at step 3, spikes to ~9 at step 4. Unlike Run 1's collapse (std_dev → 0), there IS variance here — but it comes from corruption randomness, not from the model learning different strategies. High std_dev with no reward improvement = noisy environment, not useful exploration.
- **train/guessed:** Drops from 0.85 → 0.65–0.80 over 5 steps. The model is guessing less, trending toward timeouts — early signs of the same safe-haven drift from Run 1.
- **train/corrupted_questions:** Oscillates 1.0–1.4 per episode. With ~7.5 questions at 15% rate, expected is ~1.1. Matches expectations — corruption is being applied correctly.
- **train/candidate_reduction:** Flat at ~19.2 (out of 20 objects). The model narrows aggressively regardless of corruption — it just narrows to the *wrong* candidate. This is the core problem: corruption poisons the filtering, not the narrowing behavior.
- **train/grad_norm:** Low (~0.2) for steps 0-3, spikes to 0.45 at step 4. The model makes large parameter updates based on noisy signal at step 4 — likely why step 5 eval degraded further.
- **train/forced_questions** and **train/disabled_attributes:** Correctly at 0 for 4a. Perturbation metrics logging correctly.

**Trajectory Analysis (6 episodes per eval, detailed):**

Failure patterns observed:

1. **Silent corruption acceptance.** Agent asks a corrupted question, gets a wrong answer, and doesn't notice. Candidate filtering proceeds based on the flipped answer, narrowing to ~1 candidate — but the wrong one. The agent guesses confidently and is wrong.

2. **No verification after questions.** The model sometimes calls `get_top_candidates` or `get_candidate_count` (~50% of episodes), but never acts on anomalies. When it receives 0 candidates or an empty list, it either guesses anyway or keeps asking pointlessly.

3. **Catastrophic loops under contradiction.** When corruption produces contradictory answers, the model falls into repeating the same 2-3 questions. Worst case: `is_food`/`is_edible` asked 8 times in alternation until hitting the 15-question limit. The SFT-memorized sequence has no escape hatch for contradictory state.

4. **GRPO degradation.** Step 5 shows more looping, more timeouts (2→5), and lower accuracy (10%→5%). GRPO reinforced fixed sequences rather than encouraging adaptive behavior. With ~5-10% correct rate during training, there's almost no positive signal for GRPO to amplify.

**Skill acquisition:**

| Skill | Step 0 | Step 5 |
|-------|--------|--------|
| 1. Tool use | 20/20 | 20/20 |
| 2. Sequencing | 18/20 | 15/20 (more timeouts) |
| 3. State awareness (calls get_candidate_count) | ~10/20 | ~10/20 |
| 4. History tracking | poor (repeating questions under corruption) | worse (more loops) |
| 5. Error detection | 0/20 | 0/20 |
| 6. Recovery | 0/20 | 0/20 |

**Interpretation:**

This is a meaningful negative result. Answer corruption at 15% is **fundamentally undetectable** by the agent because the corruption is invisible — the environment state is self-consistent after the flip. Detecting corruption would require the model to reason: "I asked `is_animal` and got `yes`, but the remaining candidates include `Desk` — that's inconsistent." That's cross-referencing world knowledge against tool outputs — exactly the kind of algorithmic reasoning our thesis says GRPO can't teach.

The 95% → 10% accuracy drop from corruption alone (before any GRPO) shows how brittle the SFT strategy is. It memorized a fixed question sequence that works perfectly when answers are correct, but has zero robustness to noise. GRPO couldn't help because:
1. The corruption is invisible to the agent (no observable signal to learn from)
2. The correct rate during training (~5-10%) provides near-zero positive signal
3. Recovery would require reasoning (detecting inconsistency), not just behavioral adaptation

**This distinguishes 4a from 4b/4c:** Answer corruption poisons the state silently. Forced bad start (4b) and attribute removal (4c) are observable — the agent sees unfamiliar candidate sets or "unknown" responses. Those are behavioral challenges the model can potentially adapt to.

**Cost:** ~$1.50 (5 GRPO steps + 2 evals)

**Next:** Proceed to 4b (forced bad start) and 4c (attribute removal) smoke tests. These test behavioral adaptation rather than reasoning — the perturbations are visible to the agent, which our thesis predicts GRPO can learn from.
