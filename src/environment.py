import json
import random
import string
import math
from typing import TypedDict, Optional, Literal, Any
from pydantic import BaseModel
import requests
import openai
from openai import AsyncOpenAI
import art
from rewards import compute_reward
from prompts import get_system_prompt

from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion_message_tool_call import Function
from openai.types.chat.chat_completion import Choice

MAX_QUESTIONS = 15

class Obj(TypedDict):
    id: str
    name: str
    attrs: dict[str, bool]

def load_catalog(objects_path: str, attributes_path: str) -> tuple[list[Obj], list[str]]:
    with open(objects_path, "r") as f:
        objects = json.load(f)
    with open(attributes_path, "r") as f:
        attributes = json.load(f)
    return objects, attributes

objects, attributes = load_catalog("data/objects.json", "data/attributes.json")
objects_by_id = {obj["id"]: obj for obj in objects}

class TwentyQuestionsEpisode(TypedDict):
    id: str
    secret_id: str
    candidates: list[str]
    questions_asked: int
    invalid_questions: int
    last_question: Optional[str]
    last_answer: Optional[Literal["yes", "no", "unknown"]]
    done: bool
    guessed_id: Optional[str]
    reward_fn: str
    qa_pairs: list[tuple[str, str]]
    prev_candidate_count: int
    attributes_listed: bool  # <--- NEW FLAG

def _rand_id(k: int = 6) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=k))

def generate_episode(objects: list[Obj], attributes: list[str], *, secret_id: Optional[str] = None, reward_fn: str = 'v5', prompt_version: str = "v5-strict") -> TwentyQuestionsEpisode:
    eid = _rand_id()
    if secret_id is None:
        secret_id = random.choice(objects)["id"]

    return {
        "id": eid,
        "secret_id": secret_id,
        "candidates": [o["id"] for o in objects],
        "questions_asked": 0,
        "invalid_questions": 0,
        "last_question": None,
        "last_answer": None,
        "done": False,
        "guessed_id": None,
        "qa_pairs": [],
        "reward_fn": reward_fn,
        "prev_candidate_count": len(objects),
        "attributes_listed": False,
    }

def render_state(ep: TwentyQuestionsEpisode, objects_by_id: dict[str, Obj]) -> str:
    cand_count = len(ep["candidates"])
    last_q = ep["last_question"] or "-"
    last_a = ep["last_answer"] or "-"
    
    if cand_count <= 10:
        sample = ep["candidates"][:cand_count]
    else:
        sample = random.sample(ep["candidates"], 5)
    
    sample_names = [objects_by_id[oid]["name"] for oid in sample]
    
    return (
        f"Candidates: {cand_count}\n"
        f"Questions: {ep['questions_asked']}/{MAX_QUESTIONS}\n"
        f"Last Q: {last_q} -> {last_a}\n"
        f"Top candidates: {', '.join(sample_names)}"
    )

def check_episode_finished(ep: TwentyQuestionsEpisode) -> bool:
    if ep["done"]: return True
    if ep["questions_asked"] >= MAX_QUESTIONS: return True
    return False

def ask_yesno(ep: TwentyQuestionsEpisode, objects_by_id: dict[str, Obj], attributes: list[str], attr_name: str) -> Literal["yes", "no", "unknown"]:
    ep["questions_asked"] += 1
    ep["last_question"] = f"attr:{attr_name}"

    if attr_name not in attributes:
        ep["invalid_questions"] += 1
        ep["last_answer"] = "unknown"
        return "unknown"

    ep["prev_candidate_count"] = len(ep["candidates"])

    secret = objects_by_id[ep["secret_id"]]
    secret_has = bool(secret["attrs"].get(attr_name, False))
    answer: Literal["yes", "no"] = "yes" if secret_has else "no"

    ep["qa_pairs"].append((attr_name, answer))

    if secret_has:
        ep["candidates"] = [oid for oid in ep["candidates"] if objects_by_id[oid]["attrs"].get(attr_name, False)]
    else:
        ep["candidates"] = [oid for oid in ep["candidates"] if not objects_by_id[oid]["attrs"].get(attr_name, False)]

    ep["last_answer"] = answer
    return answer

def submit_guess(ep: TwentyQuestionsEpisode, object_id: str) -> bool:
    ep["done"] = True
    ep["guessed_id"] = object_id
    return object_id == ep["secret_id"]

def tool_schemas():
    return [
        {"type": "function", "function": {"name": "list_attributes", "description": "List attributes.", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "ask_yesno", "description": "Ask attr.", "parameters": {"type": "object", "properties": {"attr_name": {"type": "string"}}, "required": ["attr_name"]}}},
        {"type": "function", "function": {"name": "get_candidate_count", "description": "Count remaining.", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "get_top_candidates", "description": "Get candidates.", "parameters": {"type": "object", "properties": {"k": {"type": "integer"}}, "required": ["k"]}}},
        {"type": "function", "function": {"name": "submit_guess", "description": "Guess object.", "parameters": {"type": "object", "properties": {"object_id": {"type": "string"}}, "required": ["object_id"]}}},
    ]

# --- HEURISTIC TEACHER LOGIC ---
def get_optimal_action(ep: TwentyQuestionsEpisode, available_attributes: list[str]) -> tuple[str, dict]:
    candidates = ep["candidates"]
    
    # 1. Win immediately if unique
    if len(candidates) == 1:
        return "submit_guess", {"object_id": candidates[0]}
    
    # 2. Force list_attributes ONLY ONCE (Fixes infinite loop)
    if not ep["attributes_listed"]:
        return "list_attributes", {}

    # 3. Determine best splitting attribute
    asked_attrs = {qa[0] for qa in ep["qa_pairs"]}
    valid_attrs = [a for a in available_attributes if a not in asked_attrs]
    
    best_attr = None
    min_diff = float('inf')
    total = len(candidates)
    
    candidate_objs = [objects_by_id[oid] for oid in candidates]

    for attr in valid_attrs:
        true_count = sum(1 for obj in candidate_objs if obj['attrs'].get(attr, False))
        diff = abs(true_count - (total - true_count))
        
        if diff < min_diff:
            min_diff = diff
            best_attr = attr
            if min_diff == 0: 
                break
            
    if best_attr:
        return "ask_yesno", {"attr_name": best_attr}
    
    return "submit_guess", {"object_id": candidates[0]}


class Scenario20Q(BaseModel):
    step: int
    secret_id: str
    reward_fn: str = "v5"
    prompt_version: str = "v4"
    use_oracle: bool = True 

@art.retry(exceptions=(requests.ReadTimeout, openai.InternalServerError, openai.APIError, openai.APIConnectionError))
async def rollout(model: art.Model, scenario: Scenario20Q) -> art.Trajectory:
    client = AsyncOpenAI(base_url=model.inference_base_url, api_key=model.inference_api_key)

    ep = generate_episode(objects, attributes, secret_id=scenario.secret_id, reward_fn=scenario.reward_fn, prompt_version=scenario.prompt_version)

    system_prompt = get_system_prompt(scenario.prompt_version)
    trajectory = art.Trajectory(
        messages_and_choices=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Find the secret object. Start by calling list_attributes()."},
        ],
        metadata={"episode_id": ep["id"], "secret_id": scenario.secret_id, "step": scenario.step},
        reward=0,
    )

    max_steps = 25

    for _ in range(max_steps):
        trajectory.messages_and_choices.append({"role": "user", "content": render_state(ep, objects_by_id)})

        if scenario.use_oracle:
            # TEACHER MODE
            tool_name, tool_args = get_optimal_action(ep, attributes)
            
            # Create REAL OpenAI Objects
            function_obj = Function(name=tool_name, arguments=json.dumps(tool_args))
            tool_call = ChatCompletionMessageToolCall(id="call_" + _rand_id(), function=function_obj, type="function")
            message = ChatCompletionMessage(role="assistant", tool_calls=[tool_call])
            choice = Choice(finish_reason="tool_calls", index=0, message=message, logprobs=None)
        else:
            # LLM MODE
            chat = await client.chat.completions.create(
                model=model.get_inference_name(),
                messages=trajectory.messages(),
                tools=tool_schemas(),
                tool_choice="auto",
                max_completion_tokens=256,
            )
            choice = chat.choices[0]

        trajectory.messages_and_choices.append(choice)
        msg = choice.message

        if not getattr(msg, "tool_calls", None):
            trajectory.messages_and_choices.append({"role": "user", "content": "Use tools."})
            continue

        for tc in msg.tool_calls:
            tool_name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")

            if tool_name == "list_attributes":
                ep["attributes_listed"] = True # <--- Flag Update
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

            trajectory.messages_and_choices.append({
                "role": "tool", "tool_call_id": tc.id, "name": tool_name, "content": json.dumps(result),
            })
            
            if tool_name in ("ask_yesno", "submit_guess"): break

        if check_episode_finished(ep): break

    r = compute_reward(ep)
    trajectory.reward = float(r)
    
    trajectory.metrics["questions_asked"] = int(ep["questions_asked"])
    trajectory.metrics["final_candidates"] = int(len(ep["candidates"]))
    trajectory.metrics["guessed"] = int(ep["guessed_id"] is not None)
    trajectory.metrics["correct"] = int(ep["guessed_id"] == ep["secret_id"]) if ep["guessed_id"] else 0
    trajectory.metrics["candidate_reduction"] = int(len(objects) - len(ep["candidates"]))

    return trajectory