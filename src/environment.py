import json
import random
import string
import math
from typing import TypedDict, Optional, Literal
from pydantic import BaseModel
import requests
from openai import AsyncOpenAI
import art
# import weave
from rewards import compute_reward

MAX_QUESTIONS = 15  # analogous to lowering WINNING_VALUE in 2048

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
    candidates: list[str]           # list of object ids
    questions_asked: int
    invalid_questions: int
    last_question: Optional[str]
    last_answer: Optional[Literal["yes", "no", "unknown"]]
    done: bool
    guessed_id: Optional[str]
    reward_fn: string


def _rand_id(k: int = 6) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=k))


def generate_episode(objects: list[Obj], attributes: list[str], *, secret_id: Optional[str] = None, reward_fn: string = 'v2') -> TwentyQuestionsEpisode:
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
        "qa_pairs": [], # lightweight q/a trace log
        "reward_fn": reward_fn
    }


def render_state(ep: TwentyQuestionsEpisode, objects_by_id: dict[str, Obj]) -> str:
    cand_count = len(ep["candidates"])
    last_q = ep["last_question"] or "-"
    last_a = ep["last_answer"] or "-"
    
    # Show top candidates if few remain
    if cand_count <= 10:
        sample = ep["candidates"][:cand_count]
    else:
        sample = random.sample(ep["candidates"], 5)
    
    sample_names = [objects_by_id[oid]["name"] for oid in sample]
    
    return (
        f"Candidates: {cand_count}\n"
        f"Questions: {ep['questions_asked']}/15\n"
        f"Last Q: {last_q} → {last_a}\n"
        f"Top candidates: {', '.join(sample_names)}"
    )


def check_episode_finished(ep: TwentyQuestionsEpisode) -> bool:
    if ep["done"]:
        return True
    if ep["questions_asked"] >= MAX_QUESTIONS:
        return True
    # optional: if only 1 candidate remains, you could auto-finish, but I suggest you don't.
    return False


def ask_yesno(
    ep: TwentyQuestionsEpisode,
    objects_by_id: dict[str, Obj],
    attributes: list[str],
    attr_name: str,
) -> Literal["yes", "no", "unknown"]:
    # Core transition: query an attribute and filter candidates deterministically
    ep["questions_asked"] += 1
    ep["last_question"] = f"attr:{attr_name}"

    if attr_name not in attributes:
        ep["invalid_questions"] += 1
        ep["last_answer"] = "unknown"
        return "unknown"

    secret = objects_by_id[ep["secret_id"]]
    secret_has = bool(secret["attrs"].get(attr_name, False))
    answer: Literal["yes", "no"] = "yes" if secret_has else "no"

    ep["qa_pairs"].append((attr_name, answer)) # update qa log


    # filter candidates
    if secret_has:
        ep["candidates"] = [
            oid for oid in ep["candidates"] if objects_by_id[oid]["attrs"].get(attr_name, False)
        ]
    else:
        ep["candidates"] = [
            oid for oid in ep["candidates"] if not objects_by_id[oid]["attrs"].get(attr_name, False)
        ]

    ep["last_answer"] = answer
    return answer


def submit_guess(ep: TwentyQuestionsEpisode, object_id: str) -> bool:
    ep["done"] = True
    ep["guessed_id"] = object_id
    return object_id == ep["secret_id"]

# tool schemas for the agent
def tool_schemas():
    return [
        {
            "type": "function",
            "function": {
                "name": "list_attributes",
                "description": "List the boolean attributes you can ask about.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ask_yesno",
                "description": "Ask if the secret object has the given attribute.",
                "parameters": {
                    "type": "object",
                    "properties": {"attr_name": {"type": "string"}},
                    "required": ["attr_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_candidate_count",
                "description": "Get how many candidate objects remain.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_top_candidates",
                "description": "Get a few remaining candidate objects (id, name).",
                "parameters": {
                    "type": "object",
                    "properties": {"k": {"type": "integer", "minimum": 1, "maximum": 50}},
                    "required": ["k"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit_guess",
                "description": "Guess the secret object by object_id.",
                "parameters": {
                    "type": "object",
                    "properties": {"object_id": {"type": "string"}},
                    "required": ["object_id"],
                },
            },
        },
    ]

class Scenario20Q(BaseModel):
    step: int
    secret_id: str
    reward_fn: str = "v2"  # Default to v2

SYSTEM_20Q = """You are playing 20 Questions to identify a secret object.

STRATEGY:
1. Start by asking about broad categories (is_animal, is_food, is_vehicle, etc.)
2. Use binary search: each question should eliminate ~half the candidates
3. When candidates < 5, use get_top_candidates to see options
4. Only submit_guess when you're confident (candidates ≤ 3 and you know the answer)

TOOLS AVAILABLE:
- list_attributes(): See all queryable attributes
- ask_yesno(attr_name): Ask if secret has this attribute (costs 0.01 reward)
- get_candidate_count(): Check how many possibilities remain
- get_top_candidates(k): See the top k remaining candidates
- submit_guess(object_id): Make your final guess (use object ID, not name)

RULES:
- Each question costs reward, so be efficient
- Invalid attribute names are heavily penalized
- Correct guess = +5 reward, wrong = -3 reward
- You have maximum 15 questions

BE STRATEGIC: Narrow down systematically before guessing!
"""

# @weave.op #DISABLE WEAVE TO SAVE COSTS
@art.retry(exceptions=(requests.ReadTimeout,))
async def rollout(model: art.Model, scenario: Scenario20Q) -> art.Trajectory:
    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )

    ep = generate_episode(
        objects, 
        attributes, 
        secret_id=scenario.secret_id,
        reward_fn=scenario.reward_fn  # Pass reward func through
    )

    trajectory = art.Trajectory(
        messages_and_choices=[
            {"role": "system", "content": SYSTEM_20Q},
            {"role": "user", "content": "Find the secret object. Start by calling list_attributes()."},
        ],
        metadata={
            "episode_id": ep["id"],
            "secret_id": scenario.secret_id,
            "step": scenario.step,
            "notebook-id": "20q",
        },
        reward=0,
    )

    max_steps = 25  # includes non-question tools; actual questions capped by MAX_QUESTIONS in env

    for _ in range(max_steps):
        # show current env state
        trajectory.messages_and_choices.append({"role": "user", "content": render_state(ep, objects_by_id)})

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

        # Encourage tool use if the model responds in plain text
        if not getattr(msg, "tool_calls", None):
            trajectory.messages_and_choices.append(
                {"role": "user", "content": "Use the tools. Do not answer in plain text."}
            )
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

            trajectory.messages_and_choices.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tool_name,
                "content": json.dumps(result),
            })

            if tool_name in ("ask_yesno", "submit_guess"):
              break

        if check_episode_finished(ep):
            break

    r = compute_reward(ep)
    trajectory.reward = float(r)

    # --- occasional debug trace (logs only, NOT metrics) ---
    if random.random() < 0.02:  # ~2% of episodes
        print(
            "TRACE",
            "secret=", scenario.secret_id,
            "q=", ep["questions_asked"],
            "cand=", len(ep["candidates"]),
            "guess=", ep["guessed_id"],
            "last=", ep["last_question"], ep["last_answer"],
            "qa=", ep["qa_pairs"][:6],
        )

    # -------- NUMERIC METRICS ONLY (safe for aggregation) --------
    trajectory.metrics["questions_asked"] = int(ep["questions_asked"])
    trajectory.metrics["invalid_questions"] = int(ep["invalid_questions"])
    trajectory.metrics["final_candidates"] = int(len(ep["candidates"]))
    trajectory.metrics["guessed"] = int(ep["guessed_id"] is not None)
    trajectory.metrics["correct"] = int(ep["guessed_id"] == ep["secret_id"]) if ep["guessed_id"] else 0

    N0 = len(objects)  # avoid hardcoding 76
    trajectory.metrics["candidate_reduction"] = int(N0 - len(ep["candidates"]))
    trajectory.metrics["candidate_reduction_frac"] = float((N0 - len(ep["candidates"])) / N0)

    # Optional: one extra numeric "signature" metric for fast-fail detection
    trajectory.metrics["asked_per_remaining"] = float(ep["questions_asked"] / max(1, len(ep["candidates"])))

    # -------- DEBUG STRINGS GO IN METADATA (won't be summed) --------
    trajectory.metadata["trace_secret"] = scenario.secret_id
    trajectory.metadata["trace_guess"] = ep["guessed_id"] or ""
    trajectory.metadata["trace_last_q"] = ep["last_question"] or ""
    trajectory.metadata["trace_last_a"] = ep["last_answer"] or ""
    trajectory.metadata["trace_first8"] = "; ".join([f"{q}={a}" for q, a in ep["qa_pairs"][:8]])

    return trajectory
