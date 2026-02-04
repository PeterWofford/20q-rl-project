AGENT_001_CONFIG = {
    "name": "20q-agent-001-v3",
    "project": "20q",
    "reward_fn": "v1",
    "prompt_version": "v1",
    "learning_rate": 1e-5,
}

AGENT_002_CONFIG = {
    "name": "20q-agent-002", 
    "project": "20q",
    "reward_fn": "v2",
    "prompt_version": "v2",
    "learning_rate": 1e-5,
}

AGENT_002_V2_CONFIG = {
    "name": "20q-agent-002-v2", 
    "project": "20q",
    "reward_fn": "v3",  # Use v3!
    "prompt_version": "v2", 
    "learning_rate": 1e-5,
}

AGENT_002_V3_CONFIG = {
    "name": "20q-agent-002-v3", 
    "project": "20q",
    "reward_fn": "v3",  # Use v3!
    "prompt_version": "v3",
    "learning_rate": 1e-5,
}

AGENT_004_CONFIG = {
    "name": "20q-agent-004", 
    "project": "20q",
    "reward_fn": "v4",  # Use v4 since v3 appeared to degenerate to timeouts
    "prompt_version": "v3",
    "learning_rate": 1e-5,
}

AGENT_005_CONFIG = {
    "name": "20q-agent-005", 
    "project": "20q",
    "reward_fn": "v5",  # try v5 to prevent suicide guessing and use new narrowing bonus
    "prompt_version": "v4", # Use v4 to help prevent attribute hallucination
    "learning_rate": 1e-5,
}

AGENT_005_V2_CONFIG = {
    "name": "20q-agent-005-v2", # run after patching indistinguishable objects and verifying data integrity
    "project": "20q",
    "reward_fn": "v5",  # try v5 to prevent suicide guessing and use new narrowing bonus
    "prompt_version": "v4", # Use v4 to help prevent attribute hallucination
    "learning_rate": 1e-5,
}

def get_agent_config(version: str):
    # Load config based on agent
    if version == "001":
        config = AGENT_001_CONFIG
    elif version == "002":
        config = AGENT_002_CONFIG
    elif version == "002-v2":
        config = AGENT_002_V2_CONFIG
    elif version == "002-v3":
        config = AGENT_002_V3_CONFIG
    elif version == "004":
        config = AGENT_004_CONFIG
    elif version == "005":
        config = AGENT_005_CONFIG
    else:
        config = AGENT_005_V2_CONFIG #default
    return config