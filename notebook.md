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
