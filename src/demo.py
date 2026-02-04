import asyncio
import random
from dotenv import load_dotenv
import art
from art.serverless.backend import ServerlessBackend

from environment import (
    generate_episode, ask_yesno, submit_guess, check_episode_finished,
    tool_schemas, objects_by_id, attributes, SYSTEM_20Q, objects, render_state
)
from openai import AsyncOpenAI
import json

load_dotenv()


async def watch_model_play(model_name: str, secret_id: str = None):
    """Watch a single episode in real-time with detailed output"""
    
    # Setup model
    model = art.TrainableModel(
        name=model_name,
        project="20q",
        base_model="OpenPipe/Qwen3-14B-Instruct",
    )
    backend = ServerlessBackend()
    await model.register(backend)
    
    # Create client
    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )
    
    # Generate episode
    if secret_id is None:
        secret_id = random.choice(objects)["id"]
    
    secret_obj = objects_by_id[secret_id]
    
    print(f"\n{'='*70}")
    print(f"🎮 WATCHING MODEL PLAY 20 QUESTIONS")
    print(f"{'='*70}")
    print(f"🎯 Secret object: {secret_obj['name']} (ID: {secret_id})")
    print(f"📝 Attributes: {', '.join([k for k, v in secret_obj['attrs'].items() if v])}")
    print(f"{'='*70}\n")
    
    ep = generate_episode(objects, attributes, secret_id=secret_id, reward_fn="v3")
    
    messages = [
        {"role": "system", "content": SYSTEM_20Q},
        {"role": "user", "content": "Find the secret object. Start by calling list_attributes()."}
    ]
    
    turn = 0
    
    for _ in range(25):
        turn += 1
        print(f"\n--- Turn {turn} ---")
        print(f"Candidates remaining: {len(ep['candidates'])}")
        
        messages.append({"role": "user", "content": render_state(ep, objects_by_id)})
        
        chat = await client.chat.completions.create(
            model=model.get_inference_name(),
            messages=messages,
            tools=tool_schemas(),
            tool_choice="auto",
            max_tokens=256,
        )
        
        choice = chat.choices[0]
        messages.append(choice.message.model_dump())
        msg = choice.message
        
        if not msg.tool_calls:
            print("❌ Model didn't use tools, prompting...")
            messages.append({"role": "user", "content": "Use the tools. Do not answer in plain text."})
            continue
        
        for tc in msg.tool_calls:
            tool_name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            
            print(f"🔧 Tool: {tool_name}")
            
            if tool_name == "list_attributes":
                result = {"attributes": attributes}
                print(f"   Listed {len(attributes)} attributes")
                
            elif tool_name == "ask_yesno":
                attr = args["attr_name"]
                ans = ask_yesno(ep, objects_by_id, attributes, attr)
                result = {"answer": ans, "candidate_count": len(ep["candidates"])}
                
                emoji = "✅" if ans == "yes" else "❌" if ans == "no" else "❓"
                print(f"   {emoji} Asked: '{attr}' → {ans}")
                print(f"   📊 Candidates now: {len(ep['candidates'])}")
                
            elif tool_name == "get_candidate_count":
                result = {"count": len(ep["candidates"])}
                print(f"   Count: {len(ep['candidates'])}")
                
            elif tool_name == "get_top_candidates":
                k = int(args.get("k", 10))
                ids = ep["candidates"][:k]
                result = {"candidates": [(oid, objects_by_id[oid]["name"]) for oid in ids]}
                print(f"   Top {k} candidates: {', '.join([objects_by_id[oid]['name'] for oid in ids[:5]])}")
                
            elif tool_name == "submit_guess":
                guessed_id = args["object_id"]
                correct = submit_guess(ep, guessed_id)
                result = {"correct": correct, "guessed_id": ep["guessed_id"]}
                
                guessed_name = objects_by_id.get(guessed_id, {}).get("name", guessed_id)
                if correct:
                    print(f"   🎉 CORRECT! Guessed: {guessed_name}")
                else:
                    print(f"   ❌ WRONG! Guessed: {guessed_name}, Secret: {secret_obj['name']}")
                
            else:
                result = {"error": f"Unknown tool: {tool_name}"}
            
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tool_name,
                "content": json.dumps(result),
            })
            
            if tool_name in ("ask_yesno", "submit_guess"):
                break
        
        if check_episode_finished(ep):
            break
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"📊 EPISODE SUMMARY")
    print(f"{'='*70}")
    print(f"Secret: {secret_obj['name']}")
    print(f"Questions asked: {ep['questions_asked']}")
    print(f"Invalid questions: {ep['invalid_questions']}")
    print(f"Final candidates: {len(ep['candidates'])}")
    
    if ep['guessed_id']:
        correct = ep['guessed_id'] == ep['secret_id']
        emoji = "🎉" if correct else "❌"
        print(f"Result: {emoji} {'CORRECT' if correct else 'WRONG'}")
    else:
        print(f"Result: ⏰ TIMEOUT (no guess)")
    print(f"{'='*70}\n")


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="20q-agent-002-v2", help="Model name to watch")
    parser.add_argument("--secret", default=None, help="Specific secret ID (optional)")
    parser.add_argument("--runs", type=int, default=1, help="Number of episodes to watch")
    args = parser.parse_args()
    
    for i in range(args.runs):
        if args.runs > 1:
            print(f"\n\n{'#'*70}")
            print(f"EPISODE {i + 1}/{args.runs}")
            print(f"{'#'*70}")
        
        await watch_model_play(args.agent, args.secret)
        
        if i < args.runs - 1:
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    asyncio.run(main())