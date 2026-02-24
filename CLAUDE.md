# CLAUDE.md — Art-20Q Experiment Runner

## PROJECT CONTEXT

This is an RL research project studying what GRPO can and can't learn. We are NOT trying to build the best 20Q agent. We are using 20Q as a controlled testbed because an oracle exists (100% accuracy, ~6.5 questions), so we can measure exactly where RL deviates from optimal.

**The thesis:** "GRPO reliably acquires behavioral patterns but cannot acquire novel algorithmic reasoning from reward signal alone. RL refines execution — it doesn't teach reasoning."

Every experiment exists to test, refine, or disprove this thesis. When analyzing results, always ask: "What does this tell us about the boundary between learnable and not-learnable through reward signal?"

## WHY 20Q IS THE RIGHT TESTBED

20Q with predefined boolean attributes is a SOLVED PROBLEM. The optimal strategy is a greedy information-gain algorithm (~30 lines of Python). An oracle exists that achieves 100% accuracy in ~6.5 questions on average across 76 objects.

We are using 20Q because:

1. The oracle provides ground truth — we can measure exactly where and how the RL agent deviates from optimal play.
2. The optimal strategy (binary search / entropy maximization) is well-defined — we can decompose it into discrete cognitive skills and check which ones RL acquires.
3. In real agent tasks (customer support, code generation, research), you don't know what optimal looks like. Here you do. That makes failure analysis precise.

If someone asks "why not just write the algorithm?" — that's the point. We're studying what RL can and can't learn, not trying to solve 20Q.

## WHY 100% IS PROBABLY UNREACHABLE

Entropy maximization is a dynamic programming problem. The correct question at each step depends on the full state of remaining candidates. GRPO learns from trajectory-level reward (the whole game), not per-step optimal action selection. It can learn "asking questions is good" and "narrowing down is good" but learning which specific attribute maximally bisects 43 remaining candidates at step 4 requires combinatorial reasoning that reward signal alone probably can't teach.

gpt-5.2's 96.7% accuracy likely comes from pretraining — it has seen binary search, decision trees, and information theory in training data. It applies a concept it already understands. A 14B model trained only via GRPO would need to DISCOVER information theory from sparse reward. That's the gap.

This distinction — applying known reasoning vs. discovering new reasoning — is the core of the thesis.

## ESTABLISHED RESULTS (Runs 1-3)

### Run 1 — GRPO from scratch: Policy collapse.
Agent learns that doing nothing or guessing immediately beats trying. Confirms: GRPO cannot learn 20Q strategy from reward signal alone.

### Run 2 — SFT on oracle trajectories: Near-optimal play.
Agent learns binary search through supervised imitation. Confirms: strategy comes from demonstration, not RL.

### Run 3 — Free-form questions (no predefined attributes): Pretraining already knows.
Base Qwen-14B asks near-optimal binary search questions out of the box. Narrows 20 candidates to ~1 in ~6 questions (log2(20) ≈ 4.3). GRPO signal was flat — nothing to teach. Candidate reduction per question was flat across training steps. Confirms: reasoning strategy comes from pretraining, not RL.

**Run 3 also revealed a finding about LLM-in-the-loop RL:** LLM-as-judge introduces a consistency tax. Judge inconsistency across objects compounds multiplicatively — 5% per-call error across 6 questions and 20 objects produces ~75% game-level error. This is a fundamental limitation: the judge must be consistent across all objects for the same question, and small per-object error rates compound across sequential questions. Deterministic environments avoid this entirely.

### Summary of Runs 1-3
Three different approaches all confirm the same half of the thesis: strategy/reasoning comes from pretraining or SFT, not from GRPO. Now we test the other half: can GRPO teach resilience?

## KNOWN FAILURE MODES

- **Policy collapse to instant guess:** Agent learns that guessing immediately (even wrong) incurs less penalty than the accumulated cost of asking questions. Reward function issue.
- **Policy collapse to timeout:** Agent learns that doing nothing is safer than trying. "Safe haven effect."
- **Silent failure with expert iteration:** Peter wrote an oracle that generated perfect trajectories and injected them. Logs showed 100% accuracy. But ART is on-policy — the oracle trajectories had no logprobs, so the optimizer skipped all gradient updates. Graphs looked green, gradients were zero.
- **LLM judge compounding noise:** In free-form mode, judge inconsistency across objects for the same question compounds multiplicatively across questions, creating an accuracy ceiling unrelated to agent quality.

These are not just bugs. They're findings about RL fragility and environment design.

## CURRENT PHASE: ERROR RECOVERY EXPERIMENTS

We've shown GRPO can't teach strategy. Can it teach resilience?

SFT-trained agents are brittle — they've only seen gold trajectories (the happy path). When something goes wrong mid-game, they have no experience recovering from non-optimal states. Real-world agents constantly operate in noisy environments where plans go sideways. If GRPO can teach recovery from errors even when it can't teach optimal strategy from scratch, that's a genuinely useful and non-obvious finding.

**The headline we're testing:** "GRPO can't teach strategy, but it can teach resilience."

### Environment

Deterministic boolean attributes. NO LLM judge. Full throughput. The 76 objects, 59 attributes, 5 tools (List_attributes, Ask_yesno, Get_candidate_count, Get_top_candidates, Submit_guess). Same as Runs 1/2.

### Base Checkpoint

SFT-trained model from Run 2 (near-optimal on clean episodes).

### Perturbation Experiments (one variable per run)

**Run 4a — Answer Corruption:**
With probability P (start at 10-15%), flip a yes/no answer when the agent calls Ask_yesno. The agent asks a valid question, gets a wrong answer. Candidate filtering proceeds based on the corrupted answer.
Tests: Can the agent detect inconsistency (e.g., candidate count doesn't shrink as expected) and adapt? Does it learn to cross-check or re-ask?

**Run 4b — Forced Bad Start:**
Pre-ask 2-3 random (non-optimal) questions before the agent takes over. The agent inherits a partially-narrowed candidate set that was NOT produced by its memorized SFT strategy.
Tests: Can the agent adapt its strategy to the actual remaining candidate set, or does it rigidly follow the memorized SFT sequence regardless of state?

**Run 4c — Attribute Removal:**
Randomly disable 10-20% of attributes each episode. When the agent tries to ask about a disabled attribute, return "unknown" or skip it. The oracle-optimal path is no longer available.
Tests: Can the agent improvise when its memorized path is blocked? Does it find alternative attributes that still narrow candidates effectively?

### Key Measurement

**Primary comparison:** SFT-only vs. SFT+GRPO accuracy on perturbed episodes.

If SFT-only drops from ~95% to ~40% under noise but SFT+GRPO holds at ~70%, that's the headline: GRPO taught resilience.

### Skill Hierarchy (updated for error recovery)

1. **Tool use** — calls Ask_yesno (trivial, should always work)
2. **Sequencing** — asks questions before guessing (trivial post-SFT)
3. **State awareness** — checks Get_candidate_count, adapts behavior based on result
4. **History tracking** — avoids re-asking attributes, notices when narrowing stalls
5. **Error detection** (NEW) — recognizes when candidate count doesn't decrease as expected after a question
6. **Recovery** (NEW) — changes strategy after detecting an anomaly (asks a verification question, tries a different attribute branch, guesses earlier when uncertain)

Skills 5-6 are behavioral patterns, not algorithmic reasoning. Our thesis predicts GRPO CAN learn these.

### What Success Looks Like

**Best outcome:** SFT+GRPO agents maintain high accuracy under perturbation while SFT-only agents crumble. Proves GRPO has a role — not as a strategy teacher, but as a robustness layer. Thesis becomes: "Use SFT/pretraining to teach the strategy. Use GRPO to make it resilient."

**Worst outcome:** SFT+GRPO agents crumble equally. Means GRPO can't even teach behavioral adaptation. Still a finding worth documenting.

### Implementation Notes

- Perturbations happen in the environment, not in the agent. The agent is not told the environment is noisy — it must discover that through experience.
- Keep the reward function simple: correct guess = big reward, fewer questions = bonus, wrong guess = penalty. Same structure as previous runs.
- The SFT checkpoint is the starting point for ALL perturbation runs. Do not retrain SFT.
- Start each perturbation type as a small smoke test (5-10 steps, 20 objects) before committing to a full run.
- Add a config flag: `perturbation_type: none | answer_corruption | forced_bad_start | attribute_removal` with `perturbation_rate: 0.15`.

## THE STACK

- **Model:** Qwen/Qwen2.5-14B-Instruct
- **RL Library:** ART (Agent Reasoning & Training) — OpenPipe's framework
- **RL Algorithm:** GRPO
- **Compute:** ART ServerlessBackend (pay-per-use, no always-on GPUs)
- **Observability:** W&B. All runs MUST be logged to the project `art-20q-runner-2026`.
- **Repo:** https://github.com/PeterWofford/20q-rl-project

## ENVIRONMENT

- 76 objects (Chess, Shark, Paris, etc.) in data/objects.json
- 59 boolean attributes in data/attributes.json
- Tools: List_attributes, Ask_yesno, Get_candidate_count, Get_top_candidates, Submit_guess
- Theoretical optimal: 100% accuracy, avg 6.5 questions (log2(76) = 6.25)

## FRONTIER BENCHMARKS

- gpt-4o-mini: 40.0% accuracy, 12.2 avg questions
- gpt-4o: 63.3%, 10.3 avg questions
- gpt-4-turbo: 73.3%, 10.3 avg questions
- gpt-5.2: 96.7%, 8.1 avg questions

## CONSTRAINTS

- Budget: <$150 total across all experiments (track spend per run)
- Compute: ServerlessBackend only, no always-on GPUs
- One variable per run — if you change two things, you learn nothing
- Annotate 20-30 trajectories per run against the skill hierarchy
- Track cost per run in a single spreadsheet

## YOUR ROLE AS EXPERIMENT RUNNER

When Peter asks you to set up, run, or analyze an experiment:

1. **Before running:** State the hypothesis clearly. What do we expect and why? What would confirm or disprove the thesis?
2. **During setup:** Minimize cost. Use small batch sizes for initial validation before longer runs. Remember the batch-size 6 vs 12 tradeoff — higher volume of noisier steps beats fewer clean steps.
3. **After results:** Analyze against the skill hierarchy first. Don't just report accuracy — report WHICH skills the agent acquired. Read actual trajectories, not just aggregate metrics.
4. **Always ask:** "Is this result telling us something about RL's capability boundaries, or is it just a hyperparameter issue?" The former is insight, the latter is tuning.
5. **Watch for silent failures.** If metrics look too good, verify that gradients are actually nonzero and that the model's behavior has actually changed. The expert iteration failure is the cautionary tale.

## LAB NOTEBOOK FORMAT

After every run, produce an entry in this format:

```
## Run [N] — [Date]
**Hypothesis:** [What I expect and why]
**Setup:** [Model, params, what changed]
**Results:** Accuracy, avg questions, failure mode, variance, cost
**Skill acquisition:** [Which of the 6 skills does the agent exhibit?]
**Interpretation:** [What does this tell us about the thesis?]
**Next:** [What to change based on this]
```

## GIT & EXECUTION GUARDRAILS (Strict)

**Remotes:**
- `private` (`PeterWofford/art-20q-experiments-2026`) — all branches and PRs go here. This is the working repo.
- `origin` (`PeterWofford/20q-rl-project`) — public repo. NEVER push branches or create PRs here.

Before executing ANY training run or experiment script (e.g., `python src/train.py`), you MUST follow this exact sequence:
1. **Commit:** Stage and commit all code changes with a descriptive message.
2. **Branch & Push:** Push the changes to a new branch on the `private` remote. NEVER push to `origin`.
3. **Merge Request:** Create a PR on the `private` repo using `gh pr create --repo PeterWofford/art-20q-experiments-2026`. NEVER create PRs on `origin`.
4. **Pause:** Ask for my explicit permission before running the training script. NEVER run a training job without this Git workflow happening first.