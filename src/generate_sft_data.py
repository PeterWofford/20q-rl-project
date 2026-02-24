"""Generate oracle SFT trajectories.

Runs the deterministic oracle over all 76 objects and writes each trajectory
as a JSONL line in the format expected by art.utils.sft.train_sft_from_file:

    {"messages": [...], "tools": [...]}

Messages use the OpenAI chat-completion dict format. Assistant messages with
tool calls are converted from Choice objects to plain dicts. Each trajectory
ends at the assistant's submit_guess tool call (before the tool response),
because ART SFT requires the last message to be an assistant turn.

Supports perturbation modes for Run 4b/4c:
  --perturbation-type forced_bad_start  (pre-ask 2-3 random questions)
  --perturbation-type attribute_removal (disable fraction of attributes)
  --perturbation-rate 0.15              (for attribute removal)
  --trajectories-per-object 3           (multiple random perturbations per object)
  --output PATH                         (override default output path)
"""

import argparse
import json
import random
import sys
import os

# Ensure src/ is on the path so sibling imports work when run as a script
sys.path.insert(0, os.path.dirname(__file__))

from environment import (
    objects,
    objects_by_id,
    attributes,
    generate_episode,
    get_optimal_action,
    check_episode_finished,
    ask_yesno,
    submit_guess,
    render_state,
    tool_schemas,
    _rand_id,
    MAX_QUESTIONS,
)
from prompts import get_system_prompt

OUTPUT_PATH = "data/sft_oracle_trajectories.jsonl"
PROMPT_VERSION = "v4"


def choice_to_dict(choice) -> dict:
    """Convert an OpenAI Choice object to a plain assistant message dict."""
    msg = choice.message
    result = {"role": "assistant"}

    if msg.content:
        result["content"] = msg.content

    if getattr(msg, "tool_calls", None):
        result["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]

    return result


def generate_oracle_trajectory(secret_id: str, perturbation_type: str = "none",
                                perturbation_rate: float = 0.0) -> list[dict]:
    """Run the oracle for one object and return a flat message list.

    For forced_bad_start: pre-asks 2-3 random questions, then oracle plays optimally.
    For attribute_removal: oracle picks best attribute from reduced set.
    """
    ep = generate_episode(
        objects, attributes, secret_id=secret_id,
        reward_fn="v5", prompt_version=PROMPT_VERSION,
        perturbation_type=perturbation_type,
        perturbation_rate=perturbation_rate,
    )

    system_prompt = get_system_prompt(PROMPT_VERSION)
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Find the secret object. Start by calling list_attributes()."},
    ]

    from openai.types.chat import (
        ChatCompletionMessage,
        ChatCompletionMessageToolCall,
    )
    from openai.types.chat.chat_completion_message_tool_call import Function
    from openai.types.chat.chat_completion import Choice

    # For attribute removal, oracle uses only available (non-disabled) attributes
    available_attrs = [a for a in attributes if a not in ep.get("disabled_attributes", [])]

    # --- Forced bad start: pre-ask 2-3 random questions ---
    if perturbation_type == "forced_bad_start":
        n_forced = random.randint(2, 3)
        asked_so_far = set()
        shuffled_attrs = list(attributes)
        random.shuffle(shuffled_attrs)
        forced_attrs = shuffled_attrs[:n_forced]

        # First, the list_attributes call (agent always sees this)
        la_tool_id = "call_" + _rand_id()
        messages.append({"role": "user", "content": render_state(ep, objects_by_id)})
        la_msg = {
            "role": "assistant",
            "tool_calls": [{"id": la_tool_id, "type": "function",
                           "function": {"name": "list_attributes", "arguments": "{}"}}],
        }
        messages.append(la_msg)
        ep["attributes_listed"] = True
        messages.append({
            "role": "tool", "tool_call_id": la_tool_id, "name": "list_attributes",
            "content": json.dumps({"attributes": attributes}),
        })

        # Pre-ask forced random questions
        for attr in forced_attrs:
            messages.append({"role": "user", "content": render_state(ep, objects_by_id)})
            tc_id = "call_" + _rand_id()
            messages.append({
                "role": "assistant",
                "tool_calls": [{"id": tc_id, "type": "function",
                               "function": {"name": "ask_yesno",
                                           "arguments": json.dumps({"attr_name": attr})}}],
            })
            ans = ask_yesno(ep, objects_by_id, attributes, attr)
            result = {"answer": ans, "candidate_count": len(ep["candidates"])}
            messages.append({
                "role": "tool", "tool_call_id": tc_id, "name": "ask_yesno",
                "content": json.dumps(result),
            })
            asked_so_far.add(attr)

        ep["forced_questions"] = n_forced

    # --- Main oracle loop ---
    max_steps = 25
    for _ in range(max_steps):
        messages.append({"role": "user", "content": render_state(ep, objects_by_id)})

        # Oracle action (uses available_attrs for attribute removal)
        tool_name, tool_args = get_optimal_action(ep, available_attrs)

        function_obj = Function(name=tool_name, arguments=json.dumps(tool_args))
        tool_call = ChatCompletionMessageToolCall(
            id="call_" + _rand_id(), function=function_obj, type="function",
        )
        message_obj = ChatCompletionMessage(role="assistant", tool_calls=[tool_call])
        choice = Choice(finish_reason="tool_calls", index=0, message=message_obj, logprobs=None)

        # Before submit_guess, inject a get_top_candidates call so the model
        # sees the actual candidate IDs (the oracle has direct access to the
        # candidate list, but the LLM needs to see IDs via tool output).
        if tool_name == "submit_guess":
            # Build get_top_candidates assistant turn
            k = len(ep["candidates"])
            gtc_args = json.dumps({"k": k})
            gtc_function = Function(name="get_top_candidates", arguments=gtc_args)
            gtc_tool_call = ChatCompletionMessageToolCall(
                id="call_" + _rand_id(), function=gtc_function, type="function",
            )
            gtc_message = ChatCompletionMessage(role="assistant", tool_calls=[gtc_tool_call])
            gtc_choice = Choice(finish_reason="tool_calls", index=0, message=gtc_message, logprobs=None)
            messages.append(choice_to_dict(gtc_choice))

            # Tool response with actual candidate IDs
            ids = ep["candidates"][:k]
            gtc_result = {"candidates": [(oid, objects_by_id[oid]["name"]) for oid in ids]}
            messages.append({
                "role": "tool",
                "tool_call_id": gtc_tool_call.id,
                "content": json.dumps(gtc_result),
            })

            # Now add the state update and submit_guess
            messages.append({"role": "user", "content": render_state(ep, objects_by_id)})
            messages.append(choice_to_dict(choice))
            break

        # Otherwise, append assistant turn + tool response and continue
        messages.append(choice_to_dict(choice))

        # Execute tool
        args = json.loads(tool_call.function.arguments or "{}")
        if tool_name == "list_attributes":
            ep["attributes_listed"] = True
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
        else:
            result = {"error": f"Unknown tool: {tool_name}"}

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result),
        })

        if check_episode_finished(ep):
            break

    return messages


def main():
    parser = argparse.ArgumentParser(description="Generate oracle SFT trajectories")
    parser.add_argument("--perturbation-type", type=str, default="none",
                        choices=["none", "forced_bad_start", "attribute_removal"],
                        help="Perturbation type for Run 4b/4c oracle data")
    parser.add_argument("--perturbation-rate", type=float, default=0.15,
                        help="Perturbation rate (for attribute_removal)")
    parser.add_argument("--trajectories-per-object", type=int, default=1,
                        help="Number of trajectories per object (>1 for perturbed data with random variation)")
    parser.add_argument("--output", type=str, default=None,
                        help="Override output path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    # Determine output path
    if args.output:
        output_path = args.output
    elif args.perturbation_type == "forced_bad_start":
        output_path = "data/sft_forced_bad_start_trajectories.jsonl"
    elif args.perturbation_type == "attribute_removal":
        output_path = "data/sft_attribute_removal_trajectories.jsonl"
    else:
        output_path = OUTPUT_PATH

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # For perturbed modes, default to multiple trajectories per object
    traj_per_obj = args.trajectories_per_object
    if args.perturbation_type != "none" and traj_per_obj == 1:
        traj_per_obj = 3  # default to 3 for perturbed data
        print(f"Auto-setting trajectories-per-object to {traj_per_obj} for {args.perturbation_type}")

    count = 0
    tools = tool_schemas()

    print(f"Generating oracle trajectories:")
    print(f"  Perturbation: {args.perturbation_type}")
    if args.perturbation_type == "attribute_removal":
        print(f"  Rate: {args.perturbation_rate}")
    print(f"  Trajectories per object: {traj_per_obj}")
    print(f"  Output: {output_path}")

    with open(output_path, "w") as f:
        for obj in objects:
            secret_id = obj["id"]
            for t in range(traj_per_obj):
                messages = generate_oracle_trajectory(
                    secret_id,
                    perturbation_type=args.perturbation_type,
                    perturbation_rate=args.perturbation_rate,
                )

                # Verify last message is assistant role
                if messages[-1]["role"] != "assistant":
                    print(f"WARNING: {obj['name']} ({secret_id}) traj {t} last message is {messages[-1]['role']}, skipping")
                    continue

                line = json.dumps({"messages": messages, "tools": tools})
                f.write(line + "\n")
                count += 1

    print(f"Wrote {count} oracle trajectories to {output_path}")

    # Spot-check: print summary for first 3
    with open(output_path, "r") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            data = json.loads(line)
            msgs = data["messages"]
            assistant_msgs = [m for m in msgs if m["role"] == "assistant"]
            tool_msgs = [m for m in msgs if m["role"] == "tool"]
            last_assistant = assistant_msgs[-1] if assistant_msgs else {}
            last_tool_call = last_assistant.get("tool_calls", [{}])[-1] if assistant_msgs else {}
            fn_name = last_tool_call.get("function", {}).get("name", "?")
            print(f"  [{i}] {len(msgs)} messages, {len(assistant_msgs)} assistant turns, "
                  f"{len(tool_msgs)} tool responses, ends with {fn_name}")


if __name__ == "__main__":
    main()
