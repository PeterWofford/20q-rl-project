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
from configs import get_agent_config

import warnings
import os

# Suppress Pydantic serialization warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

# Load environment variables
load_dotenv()
random.seed(42)

# At the top of train.py
training_stats = {
    "attempted": 0,
    "successful": 0,
    "failed": 0,
    "failure_reasons": {}
}
import json
from datetime import datetime

training_stats = {
    "attempted": 0,
    "successful": 0,
    "failed": 0,
    "failure_reasons": {},
    "steps": []
}

STATS_FILE = "training_stats.jsonl"


def log_step_result(step: int, success: bool, error: str = None):
    """Log training step result to file"""
    result = {
        "timestamp": datetime.now().isoformat(),
        "step": step,
        "success": success,
        "error": error,
        "cumulative_attempted": training_stats["attempted"],
        "cumulative_successful": training_stats["successful"],
        "cumulative_failed": training_stats["failed"],
        "success_rate": training_stats["successful"] / training_stats["attempted"] if training_stats["attempted"] > 0 else 0
    }
    
    with open(STATS_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")
    
    return result


async def train_with_retry(backend, model, train_groups, config, step_num: int, max_retries=5):
    """Train with exponential backoff on 502/524 errors"""
    training_stats["attempted"] += 1
    
    for attempt in range(max_retries):
        try:
            result = await backend.train(
                model, 
                train_groups, 
                learning_rate=config["learning_rate"]
            )
            await model.log(train_groups, metrics=result.metrics, step=result.step, split='train')
            
            # Success!
            training_stats["successful"] += 1
            log_step_result(step_num, success=True)
            print(f"✅ Step {step_num} complete!")
            return result
            
        except Exception as e:
            error_str = str(e)
            
            # Classify error
            if "524" in error_str or "timeout" in error_str.lower():
                error_type = "524_timeout"
            elif "502" in error_str or "Bad Gateway" in error_str or "gateway" in error_str.lower():
                error_type = "502_bad_gateway"
            elif "500" in error_str or "Internal server error" in error_str:
                error_type = "500_internal_error"
            elif "Connection" in error_str or "connection" in error_str.lower():
                error_type = "connection_error"
            else:
                error_type = "other"

            # Check if retryable
            is_retryable = error_type in ["524_timeout", "502_bad_gateway", "500_internal_error", "connection_error"]
            
            if is_retryable and attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 1s, 2s, 4s, 8s, 16s
                print(f"⚠️  {error_type} on attempt {attempt + 1}/{max_retries}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                # Final failure
                training_stats["failed"] += 1
                training_stats["failure_reasons"][error_type] = training_stats["failure_reasons"].get(error_type, 0) + 1
                log_step_result(step_num, success=False, error=error_type)
                print(f"❌ Step {step_num} failed after {attempt + 1} attempts: {error_type}")
                raise


def print_training_summary():
    """Print cumulative training statistics"""
    print(f"\n{'='*60}")
    print(f"📊 TRAINING STATISTICS")
    print(f"{'='*60}")
    total = training_stats["attempted"]
    successful = training_stats["successful"]
    failed = training_stats["failed"]
    
    print(f"Total attempted: {total}")
    print(f"Successful: {successful} ({successful/total*100:.1f}%)")
    print(f"Failed: {failed} ({failed/total*100:.1f}%)")
    print(f"\nFailure breakdown:")
    for reason, count in sorted(training_stats["failure_reasons"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {reason}: {count} ({count/failed*100:.1f}% of failures)")
    print(f"{'='*60}\n")


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
    parser.add_argument("--agent", default="005-v2", choices=["001", "002", "002-v2", "002-v3", "004", "005", "005-v2"])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--eval-every", type=int, default=25, help="Run eval every N steps")
    args = parser.parse_args()
    
    config = get_agent_config(args.agent)

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
                            reward_fn=config["reward_fn"],
                            prompt_version=config["prompt_version"]
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
            await train_with_retry(backend, model, train_groups, config, step_num=i+1)
            print(f"✅ Step {i + 1} complete!")
        except Exception as e:
            print(f"❌ Step {i + 1} failed after retries: {e}")
            print("Continuing to next step...")
            continue

        # Print summary every 10 steps
        if (i + 1) % 10 == 0:
            print_training_summary()
        
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
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"🎉 Training complete! Finished steps {start_step} to {end_step}")
    print(f"{'='*60}")
    print_training_summary()
    
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