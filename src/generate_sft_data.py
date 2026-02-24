"""Generate oracle SFT trajectories for Run 2.

Runs the deterministic oracle over all 76 objects and writes each trajectory
as a JSONL line in the format expected by art.utils.sft.train_sft_from_file:

    {"messages": [...], "tools": [...]}

Messages use the OpenAI chat-completion dict format. Assistant messages with
tool calls are converted from Choice objects to plain dicts. Each trajectory
ends at the assistant's submit_guess tool call (before the tool response),
because ART SFT requires the last message to be an assistant turn.
"""

import json
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


def generate_oracle_trajectory(secret_id: str) -> list[dict]:
    """Run the oracle for one object and return a flat message list."""
    ep = generate_episode(
        objects, attributes, secret_id=secret_id,
        reward_fn="v5", prompt_version=PROMPT_VERSION,
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

    max_steps = 25
    for _ in range(max_steps):
        messages.append({"role": "user", "content": render_state(ep, objects_by_id)})

        # Oracle action
        tool_name, tool_args = get_optimal_action(ep, attributes)

        function_obj = Function(name=tool_name, arguments=json.dumps(tool_args))
        tool_call = ChatCompletionMessageToolCall(
            id="call_" + _rand_id(), function=function_obj, type="function",
        )
        message_obj = ChatCompletionMessage(role="assistant", tool_calls=[tool_call])
        choice = Choice(finish_reason="tool_calls", index=0, message=message_obj, logprobs=None)

        # If this is submit_guess, append assistant turn and STOP (don't add tool response)
        if tool_name == "submit_guess":
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
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    count = 0
    tools = tool_schemas()

    with open(OUTPUT_PATH, "w") as f:
        for obj in objects:
            secret_id = obj["id"]
            messages = generate_oracle_trajectory(secret_id)

            # Verify last message is assistant role
            if messages[-1]["role"] != "assistant":
                print(f"WARNING: {obj['name']} ({secret_id}) last message is {messages[-1]['role']}, skipping")
                continue

            line = json.dumps({"messages": messages, "tools": tools})
            f.write(line + "\n")
            count += 1

    print(f"Wrote {count} oracle trajectories to {OUTPUT_PATH}")

    # Spot-check: print summary for first 3
    with open(OUTPUT_PATH, "r") as f:
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
