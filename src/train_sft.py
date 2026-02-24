"""Run 2 — Phase 2: SFT training on oracle trajectories.

Trains the base model on 76 oracle trajectories using ART's dedicated SFT path
(cross-entropy loss on assistant tokens). This avoids the silent failure from
the previous attempt where oracle trajectories had no logprobs.

After SFT, runs a 20-episode eval to verify the model actually learned something.
"""

import asyncio
import os
import sys
import random

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

import art
from art.serverless.backend import ServerlessBackend
from art.utils.sft import train_sft_from_file

SFT_DATA_PATH = "data/sft_oracle_trajectories.jsonl"
MODEL_NAME = "run2-sft-v2"
PROJECT = "art-20q-runner-2026"
BASE_MODEL = "OpenPipe/Qwen3-14B-Instruct"


async def run_sft():
    print("=== SFT Training on Oracle Trajectories ===")
    print(f"  Data: {SFT_DATA_PATH}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Base: {BASE_MODEL}")

    # Verify data exists
    if not os.path.exists(SFT_DATA_PATH):
        print(f"ERROR: {SFT_DATA_PATH} not found. Run generate_sft_data.py first.")
        sys.exit(1)

    # Count lines
    with open(SFT_DATA_PATH, "r") as f:
        n_lines = sum(1 for line in f if line.strip())
    print(f"  Trajectories: {n_lines}")

    model = art.TrainableModel(
        name=MODEL_NAME,
        project=PROJECT,
        base_model=BASE_MODEL,
    )

    backend = ServerlessBackend()
    await model.register(backend)

    step_before = await model.get_step()
    print(f"  Model step before SFT: {step_before}")

    print("\nStarting SFT training...")
    await train_sft_from_file(
        model=model,
        file_path=SFT_DATA_PATH,
        epochs=3,
        batch_size=2,
        peak_lr=2e-4,
        schedule_type="cosine",
        warmup_ratio=0.1,
        verbose=True,
    )

    step_after = await model.get_step()
    print(f"\nSFT complete. Model step after: {step_after}")

    if step_after == step_before:
        print("WARNING: Model step did not advance. SFT may have silently failed!")
    else:
        print(f"  Steps advanced: {step_after - step_before}")

    # --- Post-SFT evaluation ---
    print("\n=== Post-SFT Evaluation (20 episodes) ===")
    from environment import rollout, Scenario20Q, objects
    from train import run_evaluation

    results = await run_evaluation(
        model_name=MODEL_NAME,
        project=PROJECT,
        reward_fn="v5",
        n_episodes=20,
        save_trajectories="trajectories/run2-sft/post_sft_eval",
        prompt_version="v4",
    )

    accuracy = results["correct"] / 20 if results else 0
    if accuracy == 0:
        print("WARNING: Post-SFT accuracy is 0%. SFT may have failed silently.")
        print("Check that gradients were nonzero and model behavior changed.")
    else:
        print(f"Post-SFT accuracy: {accuracy:.0%} — SFT appears to have worked.")


if __name__ == "__main__":
    os.makedirs("trajectories/run2-sft", exist_ok=True)
    # Don't call wandb.init ourselves — ART manages its own wandb session
    # for artifact uploads using the model's project field. Calling wandb.init
    # with a different project causes artifact lookup failures.
    asyncio.run(run_sft())
