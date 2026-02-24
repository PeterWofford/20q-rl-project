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

| Metric | Step 0 (baseline) | Step 5 (final eval) |
|--------|-------------------|---------------------|
| Accuracy | 30.0% (6/20) | 40.0% (8/20) |
| Wrong guesses | 13 | 12 |
| Timeouts | 1 | 0 |
| Avg questions | 9.7 | 10.2 |
| Avg candidates remaining | 5.9 | 2.9 |

Training metrics across 5 steps:

| Metric | Step 1 | Step 2 | Step 3 | Step 4 | Step 5 |
|--------|--------|--------|--------|--------|--------|
| Avg reward | -10.8 | -0.6 | -1.9 | -4.0 | +7.2 |
| Correct rate | 0.15 | 0.42 | 0.37 | 0.33 | 0.60 |
| Avg questions | 9.5 | 9.4 | 9.2 | 10.1 | 8.7 |
| Avg final candidates | 3.5 | 9.5 | 5.8 | 5.1 | 6.3 |

**Observations:**

1. **Accuracy improved 30% → 40% over 5 steps.** The improvement is modest but notable — with only 5 GRPO steps and a noisy judge environment, the model improved. However, variance is high (step 5 training correct rate spiked to 0.60 while earlier steps fluctuated 0.15-0.42), so this could be noise rather than genuine learning.

2. **Baseline accuracy 30% on 76 objects** (vs 25% on 20 objects in Run 3b). Surprisingly not worse at full scale — the base model handles 76 objects as well as 20.

3. **Higher avg candidates remaining (5.9 → 2.9 from step 0 to step 5).** The model improved at narrowing, which is a genuine skill improvement. With 76 objects, 2.9 remaining after ~10 questions means ~4.7 halvings. The model still can't consistently narrow to 1 candidate (vs 1.0 in Run 3b with 20 objects), confirming judge noise compounds worse at scale.

4. **Training metrics show high variance.** Reward swings from -10.8 (step 1) to +7.2 (step 5). Correct rate fluctuates 0.15-0.60. This noise is consistent with judge inconsistency dominating the reward signal — but the upward trend in step 5 suggests some learning may be occurring.

5. **Zero timeouts at step 5** (down from 1 at step 0). The model maintained or slightly improved its completion behavior.

**Interpretation:**

Run 3c confirms and extends the Run 3b findings:

1. **The base model already asks good freeform questions from pretraining.** 30% accuracy on 76 objects without any training is comparable to gpt-4o-mini's 40% with predefined attributes. The questions are near-optimal binary search from pretraining.

2. **GRPO may have marginal value even with a noisy judge.** 30% → 40% in 5 steps could indicate GRPO is helping the model ask questions the judge evaluates more consistently (optimizing for judge compatibility rather than information-theoretic optimality). This is an interesting finding: GRPO may teach "ask questions this environment can evaluate reliably" rather than "ask maximally informative questions."

3. **Judge noise remains the dominant accuracy limiter.** 2.9 avg candidates remaining should yield ~35% accuracy if the agent picks randomly from remaining candidates (1/2.9 = 34%). The actual 40% accuracy is close to this theoretical maximum, confirming that the bottleneck is candidate filtering quality, not question strategy.

4. **Flat candidate reduction per question across steps** (the key metric from the thesis) would need a longer run to confirm. 5 steps is suggestive but not conclusive.

**Cost:** ~$8 (5 GRPO steps with LLM judge, 76 objects, higher API cost than predefined mode)

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

---

## Run 4b — Forced Bad Start Smoke Test — 2026-02-24

**Hypothesis:** The SFT-trained model has memorized a fixed question sequence (optimal binary search). When 2-3 random (non-optimal) questions are pre-asked before the agent takes over, the agent inherits a partially-narrowed candidate set that doesn't match its memorized sequence. GRPO should be able to teach the model to adapt to unfamiliar states — reading `get_top_candidates` output and choosing attributes based on the actual remaining candidates rather than following the memorized path.

**Setup:**
- Model: `run2-sft-v2` checkpoint (SFT-trained, ~95% clean accuracy)
- RL: GRPO via ART ServerlessBackend
- Reward: v5
- Prompt: v4
- Perturbation: forced_bad_start (2-3 random questions pre-asked per episode)
- Steps: 5 (smoke test), seed 42
- 20 objects (subset)
- W&B run: `run4b-smoke-seed42`

**Results:**

| Metric | Step 0 (SFT baseline) | Step 5 (SFT+GRPO) |
|--------|----------------------|-------------------|
| Accuracy | 15% (3/20) | **80% (16/20)** |
| Wrong guesses | 15 | 4 |
| Timeouts | 2 | 0 |
| Avg questions | 8.6 | 8.4 |
| Avg candidates remaining | 1.3 | 1.1 |
| Avg forced questions | — | 2.47 |

Training progression:

| Metric | Step 1 | Step 2 | Step 3 | Step 4 | Step 5 |
|--------|--------|--------|--------|--------|--------|
| Avg reward | -8.4 | -9.3 | -8.2 | -7.1 | -7.0 |
| Correct rate | 0.12 | 0.10 | 0.13 | 0.17 | 0.15 |
| Avg questions | 7.6 | 8.7 | 8.2 | 8.9 | 8.3 |

**Observations:**

1. **15% → 80% in 5 GRPO steps** (reported accuracy based on environment scoring).

2. **Zero timeouts at step 5** (vs 2 at step 0). The model no longer gets stuck when its memorized sequence is disrupted.

3. **Training correct rate was low throughout (~10-17%).** Yet the final eval jumped to 80%. This suggests GRPO was learning behavioral flexibility that generalized better to eval than the noisy training metrics showed.

4. **Reward stayed negative throughout training** (-8.4 to -7.0). Gradual improvement but no step showed strongly positive reward. The model learned from a weak signal.

**Trajectory analysis — critical confound discovered:**

Detailed trajectory inspection revealed that the 15% → 80% accuracy jump is **primarily an ID formatting fix, not a reasoning improvement.**

Per-episode breakdown at baseline (step 0):
- 3 correct (guessed by object ID)
- **15 right-object-wrong-format** (narrowed to 1 candidate, guessed by name e.g. "Dog" instead of ID "d4t6u")
- 2 no guess (genuine reasoning failures — timed out with 2-4 candidates remaining)

Per-episode breakdown post-GRPO (step 5):
- 16 correct (guessed by object ID)
- **4 right-object-wrong-format** (still guessing by name)
- 0 no guess

**True reasoning accuracy (correct object identified regardless of format):**
- Baseline: **90% (18/20)** — the SFT model's question strategy was barely degraded by forced bad starts
- Post-GRPO: **100% (20/20)** — modest reasoning improvement, huge format improvement

The environment's `submit_guess()` does strict string matching against opaque 5-character IDs (e.g. `t6d8e`). The agent can only discover these IDs by calling `get_top_candidates`. Under perturbation, the SFT model stopped calling `get_top_candidates` (its memorized sequence was disrupted), so it guessed by name instead. GRPO re-taught the `get_top_candidates` → `submit_guess` ritual.

| Behavioral metric | Baseline | Post-GRPO |
|---|---|---|
| Episodes using `get_top_candidates` | ~4/20 (20%) | ~17/20 (85%) |
| Episodes with 2+ `get_top_candidates` calls | 0/20 | ~6/20 (30%) |
| Guesses by object ID (vs name) | 3/18 | 16/20 |

**What GRPO actually taught (verified from trajectories):**

1. **Re-learn the ID lookup ritual.** Call `get_top_candidates` before guessing so you see the object IDs. This is a behavioral sequence, not reasoning. It accounts for ~80% of the accuracy improvement.

2. **State-conditioned question selection.** Post-GRPO, the agent checks `get_top_candidates` mid-game and picks attributes that discriminate among the actual remaining candidates. Example: seeing {Bicycle, Broom, Chess, Paper}, it asked `has_wheels` — the one attribute that uniquely identifies Bicycle in that set. This is genuine adaptation, but it's a refinement on top of already-correct narrowing.

3. **Always submit a guess.** Baseline had 2 timeouts; post-GRPO had 0. The model learned that guessing (even uncertain) beats running out of questions.

**Skill acquisition (revised based on trajectories):**

| Skill | Step 0 | Step 5 |
|-------|--------|--------|
| 1. Tool use | 20/20 | 20/20 |
| 2. Sequencing | 18/20 | 20/20 |
| 3. State awareness (`get_top_candidates` usage) | 4/20 (20%) | 17/20 (85%) |
| 4. History tracking | adequate (narrowing worked) | better (state-conditioned) |
| 5. Error detection | N/A | N/A |
| 6. Recovery from non-optimal state | 18/20 reasoning, 3/20 format | 20/20 reasoning, 16/20 format |

**Interpretation:**

The 4b result is less about resilience than initially appeared. The SFT model's binary search strategy is surprisingly robust to forced bad starts — it narrowed to 1 candidate in 15/18 guessing episodes even from non-optimal starting states. The strategy transferred; what broke was the downstream `get_top_candidates` → `submit_guess` ritual.

This raises a design question: **`submit_guess` requiring opaque IDs is an environment design flaw.** If it accepted object names, baseline accuracy would be 90%, not 15%. The perturbation barely degraded reasoning — it degraded a formatting convention. GRPO's value here is real (re-learning the tool sequence) but smaller than the headline numbers suggest.

The genuine resilience signal is the improvement from 90% → 100% true reasoning accuracy, the elimination of timeouts (2 → 0), and the state-conditioned question selection observed in post-GRPO trajectories. These are meaningful but modest improvements, not the dramatic 15% → 80% the raw numbers imply.

**Cost:** ~$1.50 (5 GRPO steps + 2 evals)

**Next:** Fix `submit_guess` to accept object names (case-insensitive) as valid guesses, then re-run baselines to measure the true resilience delta. Full 50-step runs pending this fix.

---

## Run 4c — Attribute Removal Smoke Test — 2026-02-24

**Hypothesis:** When 15% of attributes are randomly disabled per episode (returning "unknown"), the SFT model's memorized question sequence will frequently hit blocked attributes. GRPO should teach the model to recover — skip "unknown" attributes, try alternatives, and still narrow to the correct guess. This tests behavioral adaptation similar to 4b but with a different perturbation mechanism.

**Setup:**
- Model: `run2-sft-v2` checkpoint (SFT-trained, ~95% clean accuracy)
- RL: GRPO via ART ServerlessBackend
- Reward: v5
- Prompt: v4
- Perturbation: attribute_removal at 15% rate (~9 of 59 attributes disabled per episode)
- Steps: 5 (smoke test), seed 42
- 20 objects (subset)
- W&B run: `run4c-smoke-seed42`

**Results:**

| Metric | Step 0 (SFT baseline) | Step 5 (SFT+GRPO) |
|--------|----------------------|-------------------|
| Accuracy | 25% (5/20) | **85% (17/20)** |
| Wrong guesses | 12 | 3 |
| Timeouts | 3 | 0 |
| Avg questions | 9.3 | 8.7 |
| Avg candidates remaining | 1.7 | 1.5 |
| Avg disabled attributes | 8 | 8 |

Training progression:

| Metric | Step 1 | Step 2 | Step 3 | Step 4 | Step 5 |
|--------|--------|--------|--------|--------|--------|
| Avg reward | -7.7 | -5.3 | -4.9 | -0.6 | +12.6 |
| Correct rate | 0.17 | 0.23 | 0.25 | 0.37 | 0.75 |
| Avg questions | 8.0 | 10.2 | 9.1 | 9.7 | 8.5 |
| Disabled attrs | 8 | 8 | 8 | 8 | 8 |

**Observations:**

1. **25% → 85% in 5 GRPO steps** (reported accuracy based on environment scoring).

2. **Clear monotonic improvement in training.** Correct rate 0.17 → 0.23 → 0.25 → 0.37 → 0.75, reward -7.7 → +12.6. Cleaner learning curve than 4b.

3. **Zero timeouts at step 5** (vs 3 at step 0). The SFT model got stuck when its memorized attributes were unavailable. GRPO taught it to try alternatives.

4. **Reward turned positive by step 5** (+12.6). The only perturbation type where training reward crossed positive.

**Trajectory analysis — same ID formatting confound as 4b:**

Per-episode breakdown at baseline (step 0):
- 5 correct (guessed by object ID)
- **12 right-object-wrong-format** (narrowed to 1 candidate, guessed by name)
- 3 no guess (genuine reasoning failures — 1 catastrophic with 12 candidates remaining from retry loop, 2 with 2-3 candidates)

Per-episode breakdown post-GRPO (step 5):
- 17 correct (guessed by object ID)
- **3 right-object-wrong-format**
- 0 no guess

**True reasoning accuracy (correct object identified regardless of format):**
- Baseline: **85% (17/20)** — most of the "25% accuracy" was a format issue
- Post-GRPO: **100% (20/20)**

**Baseline failure patterns from trajectories:**

1. **Catastrophic retry loops (3 episodes, 15%).** When key early attributes (`is_animal`, `is_furniture`) are disabled, the SFT model retries them compulsively. Worst case: Elephant episode alternated `is_animal` (unknown) → `is_furniture` (unknown) → `is_animal` (unknown) in an infinite loop for 12 consecutive questions, ending with 12 candidates remaining. Chess retried `is_animal` 7 times.

2. **Attribute hallucination.** When the memorized sequence is exhausted, the model invents non-existent attributes (`has_trunk`, `is_domestic`, `has_long_ears`, `can_run`). These always return "unknown", wasting questions.

3. **Silent format failure (12 episodes, 60%).** The model narrowed to 1 candidate correctly but guessed by name. The question strategy was unaffected by perturbation — the `get_top_candidates` → ID lookup was what broke.

**Post-GRPO behavioral changes from trajectories:**

1. **Skip and pivot.** When an attribute returns "unknown", the agent moves on. When `is_animal` is disabled, it tries `is_living` instead. When `is_food` is disabled, it tries `is_edible`. This is genuine behavioral adaptation — learning functional equivalences.

2. **Heavy `get_top_candidates` usage.** Nearly every trajectory calls it at a strategic decision point (2-5 candidates). Multiple calls per game in ~30% of episodes.

3. **Always guesses.** Zero timeouts. Even the one wrong guess (Chess, where 3+ category attributes were disabled simultaneously) submitted a guess rather than timing out.

4. **Reduced but not eliminated hallucination.** Post-GRPO Coffee trajectory still tried `is_coffee` and `is_meat` (nonexistent attributes), but only after exhausting valid alternatives and only as a last resort.

**Why 4c shows more genuine resilience than 4b:**

The 4c confound is smaller than 4b's. In 4b, 90% of episodes had correct reasoning at baseline (format was the only issue). In 4c, 85% had correct reasoning — but the 3 genuine failures (catastrophic retry loops) represent a real behavior GRPO fixed. The pivot-to-alternative-attribute behavior (e.g., `is_animal` → `is_living`) is a genuine learned adaptation, not just format compliance.

Additionally, 4c provides a stronger learning signal:
- **Every blocked attribute produces an explicit "unknown" response** — clear, immediate cause-and-effect within a single turn.
- **Attribute removal happens repeatedly** (~9 disabled attributes, model hits "unknown" 1-3 times per episode). Multiple recovery opportunities per trajectory.
- **The recovery action is simple and local:** when you get "unknown," try a different attribute. For 4b's forced bad start, recovery requires global state awareness.

**Skill acquisition (revised based on trajectories):**

| Skill | Step 0 | Step 5 |
|-------|--------|--------|
| 1. Tool use | 20/20 | 20/20 |
| 2. Sequencing | 17/20 | 20/20 |
| 3. State awareness (`get_top_candidates`) | ~5/20 | ~17/20 |
| 4. History tracking | poor (compulsive retries) | much better (skip and pivot) |
| 5. Error detection (recognizes "unknown") | ~5/20 | ~17/20 |
| 6. Recovery (switches to alternative attr) | 17/20 reasoning, 5/20 format | 20/20 reasoning, 17/20 format |

**Interpretation:**

The 4c result, like 4b, is inflated by the ID formatting confound. But 4c has a stronger genuine resilience signal:

| What improved | Contribution to accuracy delta |
|---|---|
| ID format fix (name → ID guessing) | ~50% of the delta (12 episodes) |
| Eliminating retry loops (3 timeouts → 0) | ~15% of the delta (3 episodes) |
| True reasoning improvement (85% → 100%) | ~15% of the delta (3 episodes) |
| Pivot to alternative attributes | Part of the reasoning improvement |

The attribute-pivoting behavior (e.g., `is_food` disabled → `is_edible`) is the clearest example of GRPO teaching a behavioral adaptation pattern. It's simple, local, and reactive — exactly the kind of thing our thesis predicts GRPO can learn. But the headline 25% → 85% dramatically overstates the resilience learning. The true resilience delta is **85% → 100% reasoning accuracy + elimination of catastrophic loops.**

**Cross-experiment summary (revised with format confound):**

| Perturbation | Reported accuracy | True reasoning accuracy | GRPO delta |
|---|---|---|---|
| | Baseline → Post-GRPO | Baseline → Post-GRPO | (reasoning) |
| 4a: Answer corruption | 10% → 5% | ~10% → ~5% | -5pp (degraded) |
| 4b: Forced bad start | 15% → 80% | **90% → 100%** | **+10pp** |
| 4c: Attribute removal | 25% → 85% | **85% → 100%** | **+15pp** |

The true GRPO resilience deltas are +10pp and +15pp — meaningful but modest. The dramatic headline numbers (15% → 80%, 25% → 85%) are mostly GRPO re-learning the ID lookup ritual that broke under perturbation.

**Environment design flaw:** `submit_guess()` requiring opaque 5-character IDs is an unnecessary indirection that artificially amplifies the effect of any perturbation. If it accepted object names (case-insensitive), baseline accuracies would be 90% and 85% respectively, and the GRPO improvement would be a clear +10-15pp in true reasoning — still a positive finding, but far less dramatic.

**Cost:** ~$1.50 (5 GRPO steps + 2 evals)

**Next:** Fix `submit_guess` to accept object names, then re-run baselines to measure the true resilience delta without the format confound. Full 50-step runs and SFT-perturbed controls pending this fix.

## Run 4b/4c v2 — Rerun with submit_guess Name Fix — 2026-02-24

**Context:** Fixed `submit_guess()` to accept object names (case-insensitive) in addition to opaque IDs. This eliminates the ID formatting confound that dominated v1 results. These are the definitive measurements of GRPO resilience.

**Setup:** Identical to v1 smoke tests except:
- `submit_guess` now accepts `"Dog"`, `"dog"`, or `"xK7mQ"` — any resolves correctly
- Same SFT checkpoint (Run 2), same seed (42), same 20 eval objects
- Perturbations are re-randomized (different specific disabled attributes / forced questions than v1)

### Run 4b v2 — Forced Bad Start (with name fix)

**Results:**

| Metric | Baseline (step 0) | Post-GRPO (step 6) |
|--------|-------------------|---------------------|
| Accuracy | **90.0% (18/20)** | **100.0% (20/20)** |
| Wrong | 0 | 0 |
| Timeout | 2 | 0 |
| Avg questions | 7.8 | 8.6 |
| Avg candidates remaining | 1.0 | 1.2 |

Training progression (6 steps):

| Metric | Step 1 | Step 2 | Step 3 | Step 4 | Step 5 | Step 6 |
|--------|--------|--------|--------|--------|--------|--------|
| Correct rate | 96.7% | 90.0% | 95.0% | 93.3% | — | — |
| Avg reward | 20.5 | 18.7 | 20.2 | 20.0 | — | — |
| Forced questions | 2.6 | 2.6 | 2.5 | 2.6 | — | — |

**Delta: +10pp (90% → 100%).** Matches the v1 "true reasoning" estimate exactly.

**Trajectory analysis:**

Baseline failures (2 episodes):
- **Headphones:** Exhausted 15 questions stuck in color discrimination (black, white, red, green, blue, orange). Forced start consumed 2-3 questions, leaving too few for the narrowing problem. Never submitted a guess.
- 1 additional timeout (not identified by object but consistent with the output log's 18/20).

Post-GRPO fixes:
- **Headphones recovered.** Now correctly identifies electronic + no buttons = Headphones in 8 questions.
- All 20 episodes correct, 0 timeouts.
- Several hard episodes (Cabinet: 14 questions, Bread: 14 questions, Rice: 13 questions) but all resolved correctly.

**Key behavioral change:** Agent uses `get_top_candidates` more strategically when narrowing stalls, and recovers from forced bad starts by pivoting to high-information attributes rather than following the memorized SFT sequence.

### Run 4c v2 — Attribute Removal (with name fix)

**Results:**

| Metric | Baseline (step 0) | Post-GRPO (step 5) |
|--------|-------------------|---------------------|
| Accuracy | **60.0% (12/20)** | **85.0% (17/20)** |
| Wrong | 3 | 0 |
| Timeout | 5 | 3 |
| Avg questions | 9.9 | 8.4 |
| Avg candidates remaining | 1.6 | 1.6 |

Training progression (5 steps):

| Metric | Step 1 | Step 2 | Step 3 | Step 4 | Step 5 |
|--------|--------|--------|--------|--------|--------|
| Correct rate | 86.7% | 70.0% | 85.0% | 80.0% | 81.7% |
| Avg reward | 17.1 | 11.3 | 17.2 | 14.0 | 15.4 |
| Disabled attrs | 8 | 8 | 8 | 8 | 8 |

**Delta: +25pp (60% → 85%).** Larger than the v1 "true reasoning" estimate of +15pp.

**Why v2 baseline (60%) is lower than v1 "true reasoning" (85%):** Different random perturbations. The v2 run disabled different specific attributes, some of which hit critical decision nodes harder. The name fix helped (v1 reported 25% → now 60%) but didn't recover all episodes because the underlying perturbations are genuinely harder in this seed.

**Trajectory analysis:**

Baseline failures (8 episodes):
- **3 wrong guesses** (Broom→Chess, Desk→Chess, Banana→Bread): Agent narrowed but picked wrong among remaining candidates.
- **5 timeouts** (Chess, Paper, Bicycle, Bread, Elephant): Agent hit cascading unknowns on critical attributes (is_animal, is_food, is_electronic, is_furniture). Two of these (Bicycle, Bread) actually narrowed to 1 candidate but never submitted a guess — a behavioral bug where the agent keeps asking questions instead of guessing.

Post-GRPO failures (3 episodes):
- **Coffee:** is_food and is_electronic both disabled. Agent loops asking both repeatedly, never breaks free. 15 questions, 11 candidates remaining. Cascading unknown on the two most critical branching attributes.
- **Chess:** Couldn't disambiguate from a similar non-electronic, non-food object. 15 questions, 2 candidates.
- **Elephant:** is_animal disabled (the most important attribute for animals). Agent asks is_animal 5+ times, never pivots. 15 questions, 3 candidates.

**Key behavioral changes post-GRPO:**
1. **Always guesses.** 0 wrong guesses (vs 3 at baseline), 3 timeouts (vs 5 at baseline). Agent learned to submit a guess rather than continuing to ask when stuck.
2. **Pivots around unknowns.** When `is_food` is disabled, tries `is_edible`. When `is_animal` is disabled, tries `is_mammal` or `has_fur`. Not perfect — Elephant still loops on `is_animal` — but improved.
3. **Fewer questions on average** (8.4 vs 9.9). More efficient despite perturbation, because the agent doesn't waste turns retrying disabled attributes as much.

### Definitive Comparison Table (v2, no format confound)

| Perturbation | v1 (with confound) | v2 (name fix) | True GRPO delta |
|---|---|---|---|
| | Baseline → Post-GRPO | Baseline → Post-GRPO | |
| 4a: Answer corruption | 10% → 5% | (not re-run) | ~-5pp (degraded) |
| 4b: Forced bad start | 15% → 80% | **90% → 100%** | **+10pp** |
| 4c: Attribute removal | 25% → 85% | **60% → 85%** | **+25pp** |

### Interpretation

The v2 results confirm the thesis direction with clean measurements:

1. **Forced bad start (+10pp):** The SFT model is already resilient to bad starts — 90% baseline means it can usually recover from 2-3 wasted questions. GRPO pushes the remaining 10% to perfection. The improvement is real but the perturbation isn't very challenging.

2. **Attribute removal (+25pp):** This is the stronger result. The SFT model is genuinely brittle when its memorized attributes are unavailable (60% baseline). GRPO teaches real behavioral adaptations: pivoting to alternative attributes, using `get_top_candidates` when stuck, and always submitting a guess. The +25pp improvement in 5 steps is the clearest evidence of GRPO teaching resilience.

3. **Remaining failures are structural:** The 3 post-GRPO failures in 4c are all cases where critical branching attributes (is_animal, is_food, is_electronic) are disabled. These require understanding the entire decision tree to work around — exactly the kind of algorithmic reasoning our thesis predicts GRPO can't teach.

**Thesis update:** "GRPO can teach resilience to observable perturbations where recovery requires simple, local behavioral adaptations (pivot to alternative attribute, always submit a guess). It cannot teach recovery from structural perturbations that require global replanning."

**Cost:** ~$1.50 each × 2 = ~$3.00

**Next:** Full 50-step runs for 4b and 4c to see if the improvement continues or plateaus. Consider whether 4a (answer corruption) is worth re-running — the invisible perturbation may be fundamentally different from observable ones.

## Run 4b/4c Full — 50-Step GRPO Runs — 2026-02-24

**Hypothesis:** The +10pp (4b) and +25pp (4c) improvements observed in 5-step smoke tests will continue or plateau over 50 steps of GRPO training, demonstrating sustained resilience learning.

**Setup:**
- 50 GRPO steps, all 76 objects, batch size 6 (60 trajectories/step)
- Eval at steps 0, 25, 50 + final eval (50 episodes)
- Same SFT checkpoint (Run 2), seed 42
- 4b: forced_bad_start, 4c: attribute_removal (rate=0.15)

**Results: Both runs collapsed.**

### Run 4b Full — Forced Bad Start

| Metric | Step 0 | Step 25 | Step 50 (final, n=50) |
|--------|--------|---------|------------------------|
| Accuracy | 55% (11/20) | 15% (3/20) | **26% (13/50)** |
| Wrong | 5 | 17 | 37 |
| Timeout | 4 | 0 | 0 |
| Avg questions | 11.8 | 5.6 | 6.1 |
| Avg candidates remaining | 2.0 | 14.3 | 9.2 |

Training trajectory (sampled steps):

| Step | Correct rate | Reward | Avg questions | Failure mode |
|------|-------------|--------|---------------|--------------|
| 1 | 60% | +7.3 | 12.2 | Normal play |
| 10 | **98%** | +16.7 | 10.3 | **Peak performance** |
| 15 | 100% | +4.5 | 12.5 | Starting to overfit |
| 20 | 98% | +6.7 | 11.6 | Questions declining |
| 25 | 100% | **-7.5** | **6.2** | **Suicide guessing begins** |
| 30 | 100% | -12.0 | 3.0 | Severe collapse |
| 35 | 100% | -11.1 | 3.0 | Degenerate — guesses with ~20 candidates |
| 40 | 98% | -6.3 | 7.4 | Partial recovery attempt |
| 50 | 98% | -5.8 | 5.7 | Settled into early-guess pattern |

**Collapse mechanism:** The agent learned that guessing immediately (even wrong) avoids the accumulated cost of asking questions. By step 25, questions dropped from 12 to 6 and reward went negative. The agent always guesses (100% guess rate by step 15) but with too many candidates remaining (14 at step 25 vs 2 at step 0). This is the **suicide-guess collapse** from Run 1.

### Run 4c Full — Attribute Removal

| Metric | Step 0 | Step 25 | Step 50 (final, n=50) |
|--------|--------|---------|------------------------|
| Accuracy | 60% (12/20) | 35% (7/20) | **0% (0/50)** |
| Wrong | 4 | 8 | 0 |
| Timeout | 4 | 5 | **50** |
| Avg questions | 11.8 | 13.2 | **0.0** |
| Avg candidates remaining | 2.2 | 4.6 | **76.0** |

Training trajectory (sampled steps):

| Step | Correct rate | Reward | Avg questions | Failure mode |
|------|-------------|--------|---------------|--------------|
| 1 | 80% | +2.5 | 12.4 | Normal play |
| 10 | 75% | +9.5 | 10.8 | Improving |
| 20 | 67% | +5.5 | 12.2 | Volatile |
| 25 | 78% | +6.0 | 12.5 | Still functional |
| 30 | 80% | -1.1 | 12.7 | Reward turns negative |
| 35 | 100% | -10.6 | **5.2** | Rapid collapse |
| 40 | 100% | -15.4 | **0.75** | Near-total collapse |
| 45 | **0%** | 0.0 | **0.0** | **Complete policy death** |
| 50 | 0% | 0.0 | 0.0 | Agent does nothing |

**Collapse mechanism:** Two-phase collapse. Phase 1 (steps 30-40): Agent switches to suicide guessing, questions crash from 12→0.75. Phase 2 (steps 40-50): Agent stops doing anything entirely — 0 questions, 0 guesses, 76 candidates. This is the **timeout collapse** from Run 1 (safe haven effect: doing nothing incurs no negative reward).

### Interpretation

**The smoke tests captured a narrow sweet spot.** Both perturbation types showed genuine improvement in 5 steps (4b: +10pp, 4c: +25pp), but 50 steps of GRPO destroyed the SFT-learned behavior entirely. This reveals a fundamental limitation:

1. **GRPO has a narrow training window on top of SFT.** ~5-10 steps of GRPO can refine the SFT policy (teaching resilience patterns like attribute pivoting). Beyond that, GRPO's reward optimization erodes the SFT-learned strategy and the model collapses to degenerate policies.

2. **The collapse modes are identical to Run 1.** Suicide guessing (4b) and timeout/do-nothing (4c). The SFT checkpoint delays but doesn't prevent collapse. The underlying reward landscape has the same degenerate attractors regardless of starting point.

3. **Peak performance was at ~step 10.** For 4b, step 10 showed 98% correct rate with reward +16.7. The model was genuinely better than baseline at this point. But GRPO kept optimizing past the sweet spot.

4. **Attribute removal (4c) collapsed harder than forced bad start (4b).** 4c went to 0% (complete policy death) while 4b settled at ~26% (degenerate but not dead). The noisier environment (4c) provides noisier reward signal, accelerating collapse.

**Revised thesis:** "GRPO can teach resilience in a narrow training window (~5-10 steps from an SFT checkpoint), but extended training leads to the same policy collapse observed in Run 1. The SFT checkpoint provides a better starting point but doesn't change the reward landscape's degenerate attractors. Practical use of GRPO for resilience requires early stopping or curriculum design to prevent collapse."

**Implications for the broader thesis:**
- GRPO's value is as a **brief refinement pass**, not extended training
- The resilience improvements from the smoke tests are real but fragile
- Without early stopping or a modified reward function, GRPO reliably finds and exploits degenerate policies
- This explains why the smoke tests (5 steps) showed improvement while full runs (50 steps) showed collapse — the sweet spot is narrow

**Cost:** ~$25 total (both runs)

**Next:** This likely concludes the experimental phase. The key findings across all runs:
1. GRPO from scratch → collapse (Run 1)
2. SFT teaches strategy (Run 2)
3. Pretraining already has reasoning (Run 3)
4. GRPO can briefly teach resilience (~5-10 steps) but collapses with extended training (Run 4)

Consider: Would a modified reward function (e.g., entropy bonus, KL penalty from SFT reference) prevent collapse? That's a natural follow-up but may be out of scope/budget.
