import wandb
import argparse
import asyncio
import random
from dotenv import load_dotenv

import art
# import weave
from art.serverless.backend import ServerlessBackend
from art.utils.strip_logprobs import strip_logprobs

from environment import rollout, Scenario20Q, objects
from configs import AGENT_001_CONFIG, AGENT_002_CONFIG, AGENT_002_V2_CONFIG, AGENT_002_V3_CONFIG

import warnings
import os

# Suppress Pydantic serialization warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

# Load environment variables
load_dotenv()
random.seed(42)

async def train_with_retry(backend, model, train_groups, config, max_retries=5):
    """Train with exponential backoff on 502 errors"""
    for attempt in range(max_retries):
        try:
            result = await backend.train(
                model, 
                train_groups, 
                learning_rate=config["learning_rate"]
            )
            await model.log(train_groups, metrics=result.metrics, step=result.step, split='train')
            return result
        except Exception as e:
            error_str = str(e)
            if "502" in error_str or "Bad Gateway" in error_str or "gateway" in error_str.lower():
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s, 8s, 16s
                print(f"⚠️  502 error on attempt {attempt + 1}/{max_retries}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
                if attempt == max_retries - 1:
                    print(f"❌ Failed after {max_retries} attempts. Moving to next step.")
                    raise
            else:
                raise


async def run_evaluation(model_name: str, project: str, reward_fn: str, n_episodes: int = 20):
    """Quick evaluation during training"""
    print(f"\n{'='*60}")
    print(f"🔍 EVALUATION at current checkpoint")
    print(f"{'='*60}")
    
    eval_model = art.TrainableModel(
        name=model_name,
        project=project,
        base_model="OpenPipe/Qwen3-14B-Instruct",
    )
    
    eval_backend = ServerlessBackend()
    await eval_model.register(eval_backend)
    
    current_step = await eval_model.get_step()
    print(f"Evaluating model at step: {current_step}")
    
    # Eval on random secrets
    eval_secrets = random.sample([o["id"] for o in objects], min(n_episodes, len(objects)))
    
    results = {
        "correct": 0,
        "wrong": 0,
        "timeout": 0,
        "total_questions": 0,
        "total_candidates": 0,
    }
    
    for secret_id in eval_secrets:
        try:
            trajectory = await rollout(
                eval_model,
                Scenario20Q(step=current_step, secret_id=secret_id, reward_fn=reward_fn)
            )
            
            if trajectory.metrics.get("correct") == 1:
                results["correct"] += 1
            elif trajectory.metrics.get("guessed") == 1:
                results["wrong"] += 1
            else:
                results["timeout"] += 1
            
            results["total_questions"] += trajectory.metrics.get("questions_asked", 0)
            results["total_candidates"] += trajectory.metrics.get("final_candidates", 0)
        except Exception as e:
            print(f"  Eval error: {e}")
            continue
    
    total = len(eval_secrets)
    accuracy = results["correct"] / total if total > 0 else 0
    avg_questions = results["total_questions"] / total if total > 0 else 0
    avg_candidates = results["total_candidates"] / total if total > 0 else 0
    
    print(f"\n📊 EVAL RESULTS (step {current_step}):")
    print(f"  Accuracy: {accuracy:.1%} ({results['correct']}/{total})")
    print(f"  Wrong: {results['wrong']}, Timeout: {results['timeout']}")
    print(f"  Avg questions: {avg_questions:.1f}")
    print(f"  Avg candidates remaining: {avg_candidates:.1f}")
    print(f"{'='*60}\n")
    
    return results


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="002-v3", choices=["001", "002", "002-v2", "002-v3"])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--eval-every", type=int, default=25, help="Run eval every N steps")
    args = parser.parse_args()
    
    # Load config based on agent
    if args.agent == "001":
        config = AGENT_001_CONFIG
    elif args.agent == "002-v2":
        config = AGENT_002_V2_CONFIG
    elif args.agent == "002-v3":
        config = AGENT_002_V3_CONFIG
    else:
        config = AGENT_002_CONFIG

    # Set consistent WandB run ID to prevent fragmentation
    os.environ["WANDB_RUN_ID"] = config["name"]
    os.environ["WANDB_RESUME"] = "allow"

    # Initialize Weave (DISABLED to save costs)
    # weave.init(
    #     config["project"],
    #     settings={"print_call_link": False},
    #     global_postprocess_output=strip_logprobs
    # )
    
    # Create model
    model = art.TrainableModel(
        name=config["name"],
        project=config["project"],
        base_model="OpenPipe/Qwen3-14B-Instruct",
    )
    
    backend = ServerlessBackend()
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
        print(f"\n=== Step {i + 1}/{end_step} ===")
        
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
                            reward_fn=config["reward_fn"]
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
        
        # Train with retry logic
        try:
            await train_with_retry(backend, model, train_groups, config)
            print(f"✅ Step {i + 1} complete!")
        except Exception as e:
            print(f"❌ Step {i + 1} failed after retries: {e}")
            print("Continuing to next step...")
            continue
        
        # Run evaluation every N steps
        if args.eval_every > 0 and (i + 1) % args.eval_every == 0:
            try:
                await run_evaluation(
                    config["name"], 
                    config["project"], 
                    config["reward_fn"],
                    n_episodes=20
                )
            except Exception as e:
                print(f"⚠️  Evaluation failed: {e}")
                print("Continuing training...")
    
    print(f"\n{'='*60}")
    print(f"🎉 Training complete! Finished steps {start_step} to {end_step}")
    print(f"{'='*60}")
    
    # Final evaluation
    print("\n🏁 Running final evaluation...")
    try:
        await run_evaluation(
            config["name"], 
            config["project"], 
            config["reward_fn"],
            n_episodes=50
        )
    except Exception as e:
        print(f"Final evaluation failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())