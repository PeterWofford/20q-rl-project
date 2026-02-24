"""Run 4b/4c — SFT on perturbed oracle trajectories.

Continues from the Run 2 SFT checkpoint (clean play) and fine-tunes on
perturbed oracle trajectories. This creates the SFT-perturbed control
for the 3-way comparison:

  1. SFT-clean (Run 2) -> perturbed eval (baseline)
  2. SFT-clean + perturbed SFT (this script) -> perturbed eval (control)
  3. SFT-clean + GRPO (train.py) -> perturbed eval (treatment)

Usage:
  python src/train_sft_perturbed.py --perturbation-type forced_bad_start
  python src/train_sft_perturbed.py --perturbation-type attribute_removal
"""

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

import art
from art.serverless.backend import ServerlessBackend
from art.utils.sft import train_sft_from_file

# The Run 2 SFT checkpoint name — we continue from this
BASE_SFT_NAME = "run2-sft-v2"
PROJECT = "art-20q-runner-2026"
BASE_MODEL = "OpenPipe/Qwen3-14B-Instruct"

# Default data paths per perturbation type
DATA_PATHS = {
    "forced_bad_start": "data/sft_forced_bad_start_trajectories.jsonl",
    "attribute_removal": "data/sft_attribute_removal_trajectories.jsonl",
}

# Model names for the perturbed SFT checkpoints
MODEL_NAMES = {
    "forced_bad_start": "run4b-sft-perturbed",
    "attribute_removal": "run4c-sft-perturbed",
}


async def run_sft_perturbed(perturbation_type: str, data_path: str = None,
                             epochs: int = 3, batch_size: int = 2):
    model_name = MODEL_NAMES[perturbation_type]
    sft_data = data_path or DATA_PATHS[perturbation_type]

    print(f"=== Perturbed SFT Training ({perturbation_type}) ===")
    print(f"  Data: {sft_data}")
    print(f"  Model: {model_name} (continuing from {BASE_SFT_NAME})")
    print(f"  Base: {BASE_MODEL}")

    if not os.path.exists(sft_data):
        print(f"ERROR: {sft_data} not found.")
        print(f"Run: python src/generate_sft_data.py --perturbation-type {perturbation_type}")
        sys.exit(1)

    with open(sft_data, "r") as f:
        n_lines = sum(1 for line in f if line.strip())
    print(f"  Trajectories: {n_lines}")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}")

    model = art.TrainableModel(
        name=model_name,
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
        file_path=sft_data,
        epochs=epochs,
        batch_size=batch_size,
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

    # Post-SFT evaluation on perturbed episodes
    print(f"\n=== Post-SFT Evaluation (20 episodes, {perturbation_type}) ===")
    from train import run_evaluation
    from configs import ExperimentConfig

    perturbation_rate = 0.15 if perturbation_type == "attribute_removal" else 0.0
    traj_dir = f"trajectories/{model_name}"
    os.makedirs(traj_dir, exist_ok=True)

    eval_config = ExperimentConfig(
        name=model_name,
        project=PROJECT,
        perturbation_type=perturbation_type,
        perturbation_rate=perturbation_rate,
    )
    results = await run_evaluation(
        eval_config,
        eval_model_name=model_name,
        n_episodes=20,
        save_trajectories=f"{traj_dir}/post_sft_eval",
    )

    accuracy = results["correct"] / 20 if results else 0
    if accuracy == 0:
        print("WARNING: Post-SFT accuracy is 0%. Check gradients and model behavior.")
    else:
        print(f"Post-SFT accuracy on perturbed episodes: {accuracy:.0%}")


def main():
    parser = argparse.ArgumentParser(description="SFT on perturbed oracle trajectories (Run 4 control)")
    parser.add_argument("--perturbation-type", type=str, required=True,
                        choices=["forced_bad_start", "attribute_removal"],
                        help="Which perturbation type to train on")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Override SFT data path")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    asyncio.run(run_sft_perturbed(
        perturbation_type=args.perturbation_type,
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
    ))


if __name__ == "__main__":
    main()
