"""Test plan for Run 3 freeform mode.

Step 1: Manual judge test — verify evaluate_question returns correct answers
Step 2: Run 5 eval episodes from base model in freeform mode
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv
load_dotenv()

from environment import (
    evaluate_question, ask_freeform, generate_episode,
    objects, objects_by_id, attributes, tool_schemas,
)


async def test_judge():
    """Step 1: Verify judge returns correct yes/no/unknown for known questions."""
    print("=" * 60)
    print("STEP 1: Manual judge test")
    print("=" * 60)

    # Find Dog and Car objects
    dog = next((o for o in objects if o["name"].lower() == "dog"), None)
    car = next((o for o in objects if o["name"].lower() == "car"), None)

    if not dog or not car:
        # Fallback: pick first two objects and use generic questions
        dog = objects[0]
        car = objects[1]
        print(f"  (Dog/Car not found, using {dog['name']} and {car['name']})")

    test_cases = [
        (dog["id"], dog["name"], dog["attrs"], "Is it an animal?", "yes"),
        (car["id"], car["name"], car["attrs"], "Is it an animal?", "no"),
        (dog["id"], dog["name"], dog["attrs"], "Is it man-made?", "no"),
        (car["id"], car["name"], car["attrs"], "Is it man-made?", "yes"),
        # Edge case: vague question
        (dog["id"], dog["name"], dog["attrs"], "Is it related to the concept of infinity?", "no"),
    ]

    passed = 0
    for obj_id, obj_name, attrs, question, expected in test_cases:
        answer = await evaluate_question(obj_id, obj_name, attrs, question)
        status = "PASS" if answer == expected else "FAIL"
        if status == "PASS":
            passed += 1
        print(f"  {status}: {obj_name} + \"{question}\" -> {answer} (expected {expected})")

    print(f"\n  {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)


async def test_freeform_episode():
    """Step 1b: Verify ask_freeform filters candidates correctly."""
    print("\n" + "=" * 60)
    print("STEP 1b: Freeform episode candidate filtering")
    print("=" * 60)

    # Pick a known animal object as secret
    animal = next((o for o in objects if o["attrs"].get("is_animal", False)), objects[0])
    print(f"  Secret: {animal['name']} ({animal['id']})")

    ep = generate_episode(objects, attributes, secret_id=animal["id"], question_mode="freeform")
    print(f"  Initial candidates: {len(ep['candidates'])}")

    answer = await ask_freeform(ep, objects_by_id, "Is it an animal?")
    print(f"  Q: 'Is it an animal?' -> {answer}")
    print(f"  Candidates after: {len(ep['candidates'])}")
    print(f"  Secret still in candidates: {animal['id'] in ep['candidates']}")
    print(f"  Questions asked: {ep['questions_asked']}")
    print(f"  Invalid questions: {ep['invalid_questions']}")

    # Candidates should have decreased significantly
    if len(ep["candidates"]) < 76 and animal["id"] in ep["candidates"]:
        print("  PASS: Candidates filtered, secret retained")
        return True
    else:
        print("  FAIL: Filtering issue")
        return False


async def test_tool_schemas():
    """Verify tool schemas differ between modes."""
    print("\n" + "=" * 60)
    print("STEP 1c: Tool schema check")
    print("=" * 60)

    predefined = tool_schemas("predefined")
    freeform = tool_schemas("freeform")

    pred_names = {t["function"]["name"] for t in predefined}
    free_names = {t["function"]["name"] for t in freeform}

    print(f"  Predefined tools: {sorted(pred_names)}")
    print(f"  Freeform tools:   {sorted(free_names)}")

    checks = [
        ("ask_yesno in predefined", "ask_yesno" in pred_names),
        ("list_attributes in predefined", "list_attributes" in pred_names),
        ("ask_question in freeform", "ask_question" in free_names),
        ("ask_yesno NOT in freeform", "ask_yesno" not in free_names),
        ("list_attributes NOT in freeform", "list_attributes" not in free_names),
        ("submit_guess in both", "submit_guess" in pred_names and "submit_guess" in free_names),
    ]

    passed = 0
    for label, ok in checks:
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"  {status}: {label}")

    print(f"\n  {passed}/{len(checks)} passed")
    return passed == len(checks)


async def test_eval_episodes():
    """Step 2: Run 5 eval episodes from base model in freeform mode."""
    print("\n" + "=" * 60)
    print("STEP 2: 5 eval episodes (base model, freeform)")
    print("=" * 60)

    import art
    from art.serverless.backend import ServerlessBackend
    from environment import rollout, Scenario20Q
    import random

    model = art.TrainableModel(
        name="run3-freeform-test",
        project="art-20q-runner-2026",
        base_model="OpenPipe/Qwen3-14B-Instruct",
    )

    backend = ServerlessBackend()
    await model.register(backend)

    eval_secrets = random.sample([o["id"] for o in objects], 5)

    for i, secret_id in enumerate(eval_secrets):
        secret_name = objects_by_id[secret_id]["name"]
        print(f"\n  Episode {i+1}/5: {secret_name}")
        try:
            trajectory = await rollout(
                model,
                Scenario20Q(
                    step=0,
                    secret_id=secret_id,
                    reward_fn="v5",
                    prompt_version="v6",
                    use_oracle=False,
                    question_mode="freeform",
                )
            )
            correct = trajectory.metrics.get("correct", 0)
            questions = trajectory.metrics.get("questions_asked", 0)
            candidates = trajectory.metrics.get("final_candidates", 0)
            reward = trajectory.reward
            print(f"    Correct: {bool(correct)} | Questions: {questions} | Candidates left: {candidates} | Reward: {reward:.1f}")

            # Print tool call sequence
            for item in trajectory.messages_and_choices:
                if hasattr(item, 'message') and getattr(item.message, 'tool_calls', None):
                    for tc in item.message.tool_calls:
                        import json
                        args = json.loads(tc.function.arguments or "{}")
                        if tc.function.name == "ask_question":
                            print(f"    -> ask_question(\"{args.get('question', '')}\")")
                        elif tc.function.name == "submit_guess":
                            print(f"    -> submit_guess({args.get('object_id', '')})")
                        else:
                            print(f"    -> {tc.function.name}({args})")
        except Exception as e:
            print(f"    ERROR: {e}")


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=0, help="0=all local tests, 1=judge only, 2=eval episodes")
    args = parser.parse_args()

    if args.step in (0, 1):
        await test_tool_schemas()
        await test_judge()
        await test_freeform_episode()

    if args.step in (0, 2):
        await test_eval_episodes()


if __name__ == "__main__":
    asyncio.run(main())
