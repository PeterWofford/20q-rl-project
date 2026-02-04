AGENT_001_CONFIG = {
    "name": "20q-agent-001-v3",
    "project": "20q",
    "reward_fn": "v1",
    "learning_rate": 1e-5,
}

AGENT_002_CONFIG = {
    "name": "20q-agent-002", 
    "project": "20q",
    "reward_fn": "v2",
    "learning_rate": 1e-5,
}

AGENT_002_V2_CONFIG = {
    "name": "20q-agent-002-v2", 
    "project": "20q",
    "reward_fn": "v3",  # Use v3!
    "learning_rate": 1e-5,
}

AGENT_002_V3_CONFIG = {
    "name": "20q-agent-002-v3", 
    "project": "20q",
    "reward_fn": "v3",  # Use v3!
    "learning_rate": 1e-5,
}

AGENT_004_CONFIG = {
    "name": "20q-agent-004", 
    "project": "20q",
    "reward_fn": "v4",  # Use v4 since v3 appeared to degenerate to timeouts
    "learning_rate": 1e-5,
}