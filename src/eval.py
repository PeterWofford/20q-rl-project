import asyncio
import random
from dotenv import load_dotenv
from openai import AsyncOpenAI
import os

from environment import Scenario20Q, objects, load_catalog, generate_episode, ask_yesno, submit_guess, check_episode_finished, tool_schemas, objects_by_id, attributes, SYSTEM_20Q
import json

load_dotenv()
random.seed(42)


async def rollout_frontier_model(model_name: str, api_key: str, secret_id: str, max_steps: int = 25):
    """Run a single episode with a frontier model"""
    client = AsyncOpenAI(api_key=api_key)
    
    ep = generate_episode(objects, attributes, secret_id=secret_id, reward_fn="v3")
    
    messages = [
        {"role": "system", "content": SYSTEM_20Q},
        {"role": "user", "content": "Find the secret object. Start by calling list_attributes()."}
    ]
    
    for _ in range(max_steps):
        messages.append({"role": "user", "content": f"Candidates remaining: {len(ep['candidates'])}"})
        
        chat = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tool_schemas(),
            tool_choice="auto",
            max_tokens=256,
        )
        
        choice = chat.choices[0]
        messages.append(choice.message.model_dump())
        
        msg = choice.message
        
        if not msg.tool_calls:
            messages.append({"role": "user", "content": "Use the tools. Do not answer in plain text."})
            continue
        
        for tc in msg.tool_calls:
            tool_name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            
            if tool_name == "list_attributes":
                result = {"attributes": attributes}
            elif tool_name == "ask_yesno":
                ans = ask_yesno(ep, objects_by_id, attributes, args["attr_name"])
                result = {"answer": ans, "candidate_count": len(ep["candidates"])}
            elif tool_name == "get_candidate_count":
                result = {"count": len(ep["candidates"])}
            elif tool_name == "get_top_candidates":
                k = int(args.get("k", 10))
                ids = ep["candidates"][:k]
                result = {"candidates": [(oid, objects_by_id[oid]["name"]) for oid in ids]}
            elif tool_name == "submit_guess":
                correct = submit_guess(ep, args["object_id"])
                result = {"correct": correct, "guessed_id": ep["guessed_id"]}
            else:
                result = {"error": f"Unknown tool: {tool_name}"}
            
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tool_name,
                "content": json.dumps(result),
            })
        
        if check_episode_finished(ep):
            break
    
    return {
        "correct": ep["guessed_id"] == ep["secret_id"] if ep["guessed_id"] else False,
        "questions_asked": ep["questions_asked"],
        "candidates_remaining": len(ep["candidates"]),
        "guessed": ep["guessed_id"] is not None,
    }


async def rollout_frontier_model_with_retry(model_name: str, api_key: str, secret_id: str, max_retries: int = 5):
    """Rollout with exponential backoff for rate limits"""
    for attempt in range(max_retries):
        try:
            return await rollout_frontier_model(model_name, api_key, secret_id, max_steps=25)
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)  # 1s, 2s, 4s, 8s, 16s + jitter
                    print(f"  Rate limit, waiting {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})...")
                    await asyncio.sleep(wait_time)
                else:
                    raise
            else:
                raise


async def evaluate_frontier_model(model_name: str, api_key: str, n_episodes: int = 30):
    """Evaluate a frontier model"""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}\n")
    
    eval_secrets = random.sample([o["id"] for o in objects], min(n_episodes, len(objects)))
    
    results = {
        "correct": 0,
        "wrong": 0,
        "timeout": 0,
        "total_questions": 0,
        "total_candidates": 0,
        "errors": 0, 
    }
    
    for i, secret_id in enumerate(eval_secrets):
        try:
            episode_result = await rollout_frontier_model_with_retry(model_name, api_key, secret_id)
            
            if episode_result["correct"]:
                results["correct"] += 1
            elif episode_result["guessed"]:
                results["wrong"] += 1
            else:
                results["timeout"] += 1
            
            results["total_questions"] += episode_result["questions_asked"]
            results["total_candidates"] += episode_result["candidates_remaining"]
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(eval_secrets)}")
        except Exception as e:
            results["errors"] += 1  # Track errors
            print(f"  Error on episode {i}: {e}")
            continue
        
        # Small delay between episodes
        await asyncio.sleep(0.3)
    
    total = results["correct"] + results["wrong"] + results["timeout"]
    actual_completed = len(eval_secrets) - results["errors"]  # Actual episodes that finished

    accuracy = results["correct"] / total if total > 0 else 0
    avg_questions = results["total_questions"] / total if total > 0 else 0
    avg_candidates = results["total_candidates"] / total if total > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"📊 RESULTS for {model_name}:")
    print(f"{'='*60}")
    print(f"Completed: {total}/{len(eval_secrets)} (errors: {results['errors']})")
    print(f"Accuracy: {accuracy:.1%} ({results['correct']}/{total})")
    print(f"Wrong: {results['wrong']}, Timeout: {results['timeout']}")
    print(f"Avg questions: {avg_questions:.1f}")
    print(f"Avg candidates remaining: {avg_candidates:.1f}")
    print(f"{'='*60}\n")
    
    return results


async def main():
    openai_key = os.getenv("OPENAI_API_KEY")
    
    # Benchmark frontier models
    models = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
    ]
    
    all_results = {}
    
    for model in models:
        results = await evaluate_frontier_model(model, openai_key, n_episodes=30)
        all_results[model] = results
    
    # Print comparison
    print(f"\n{'='*60}")
    print("🏆 FRONTIER MODEL COMPARISON")
    print(f"{'='*60}")
    for model, results in all_results.items():
        total = results["correct"] + results["wrong"] + results["timeout"]
        acc = results["correct"] / total if total > 0 else 0
        print(f"{model:20} | Accuracy: {acc:5.1%} | Avg Q: {results['total_questions']/total:4.1f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())