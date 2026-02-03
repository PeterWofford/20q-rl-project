import os
import warnings
from dotenv import load_dotenv

# 1. Load env vars (API keys)
load_dotenv()

# 2. FORCE OFFLINE MODE
# We set this in Python to ensure the script never accidentally runs online,
# preventing network timeouts on your rented A100.
os.environ["WANDB_MODE"] = "offline" 
os.environ["WANDB_SILENT"] = "true"
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

import argparse
import asyncio
import random

import art
import weave
from art.local import LocalBackend
from art.utils.strip_logprobs import strip_logprobs

from environment import rollout, Scenario20Q, objects
from configs import AGENT_001_CONFIG, AGENT_002_CONFIG, AGENT_002_V2_CONFIG

# Suppress Pydantic serialization warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

random.seed(42)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="002", choices=["001", "002", "002-v2"])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=12)
    args = parser.parse_args()
    
    # Load config based on agent
    if args.agent == "001":
        config = AGENT_001_CONFIG
    elif args.agent == "002-v2":
        config = AGENT_002_V2_CONFIG
    else:
        config = AGENT_002_CONFIG

    # Set consistent WandB run ID to prevent fragmentation
    os.environ["WANDB_RUN_ID"] = config["name"]  # Use model name as run ID
    os.environ["WANDB_RESUME"] = "allow"         # Resume if exists, create if not

    # Initialize Weave
    weave.init(
        config["project"],
        settings={"print_call_link": False},
        global_postprocess_output=strip_logprobs
    )
    
    # Create model
    model = art.TrainableModel(
        name=config["name"],
        project=config["project"],
        base_model="OpenPipe/Qwen3-14B-Instruct",
    )
    
    backend = LocalBackend()
    await model.register(backend)
    
    # Prepare secrets for training
    secrets = [o["id"] for o in objects]
    random.shuffle(secrets)
    
    BATCH = args.batch_size
    N = len(secrets)
    
    start_step = await model.get_step()
    end_step = start_step + args.steps
    print(f"Starting training from step {start_step} to {end_step}")
    
    for i in range(start_step, end_step):
        print(f"\n=== Step {i}/{end_step} ===")
        
        # Cycle through secrets
        start = (i * BATCH) % N
        step_secrets = secrets[start : start + BATCH]
        if len(step_secrets) < BATCH:
            step_secrets += secrets[: (BATCH - len(step_secrets))]
        
        print(f"Gathering trajectories for {len(step_secrets)} secrets...")
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout(
                        model, 
                        Scenario20Q(
                            step=i, 
                            secret_id=sid,
                            reward_fn=config["reward_fn"]  # Pass from config
                        )
                    ) 
                    for _ in range(10)
                )
                for sid in step_secrets
            ),
            pbar_desc="gather",
            max_exceptions=60,
        )
        
        print(f"Training on step {i}...")
        await model.delete_checkpoints("train/reward")
        result = await backend.train(
            model, 
            train_groups, 
            learning_rate=config["learning_rate"]
        )
        await model.log(train_groups, metrics=result.metrics, step=result.step, split='train')
                
        print(f"Step {i} complete!")
    
    print(f"\n=== Training complete! Finished steps {start_step} to {end_step} ===")


if __name__ == "__main__":
    asyncio.run(main())