from dataclasses import dataclass
from typing import Optional


@dataclass
class ExperimentConfig:
    name: str
    project: str
    reward_fn: str = "v5"
    prompt_version: str = "v4"
    learning_rate: float = 1e-5
    question_mode: str = "predefined"
    perturbation_type: str = "none"
    perturbation_rate: float = 0.0
    # Run 5 GRPO fields
    ppo: bool = False
    epsilon: Optional[float] = None
    epsilon_high: Optional[float] = None
    beta: float = 0.0
    scale_rewards: bool = True
    group_size: int = 10


AGENT_001_CONFIG = ExperimentConfig(
    name="20q-agent-001-v3",
    project="20q",
    reward_fn="v1",
    prompt_version="v1",
)

AGENT_002_CONFIG = ExperimentConfig(
    name="20q-agent-002",
    project="20q",
    reward_fn="v2",
    prompt_version="v2",
)

AGENT_002_V2_CONFIG = ExperimentConfig(
    name="20q-agent-002-v2",
    project="20q",
    reward_fn="v3",  # Use v3!
    prompt_version="v2",
)

AGENT_002_V3_CONFIG = ExperimentConfig(
    name="20q-agent-002-v3",
    project="20q",
    reward_fn="v3",  # Use v3!
    prompt_version="v3",
)

AGENT_004_CONFIG = ExperimentConfig(
    name="20q-agent-004",
    project="20q",
    reward_fn="v4",  # Use v4 since v3 appeared to degenerate to timeouts
    prompt_version="v3",
)

AGENT_005_CONFIG = ExperimentConfig(
    name="20q-agent-005",
    project="20q",
    reward_fn="v5",  # try v5 to prevent suicide guessing and use new narrowing bonus
    prompt_version="v4",  # Use v4 to help prevent attribute hallucination
)

AGENT_005_V2_CONFIG = ExperimentConfig(
    name="20q-agent-005-v2",  # run after patching indistinguishable objects and verifying data integrity
    project="20q",
    reward_fn="v5",  # try v5 to prevent suicide guessing and use new narrowing bonus
    prompt_version="v4",  # Use v4 to help prevent attribute hallucination
)

AGENT_006_CONFIG = ExperimentConfig(
    name="20q-agent-007-oracle",
    project="20q",
    reward_fn="v5",  # High stakes reward (doesn't matter much since Oracle is perfect)
    prompt_version="v4",
)

AGENT_RUN2_CONFIG = ExperimentConfig(
    name="run2-sft-v2",  # Same name as SFT model so GRPO continues from SFT checkpoint
    project="art-20q-runner-2026",
    reward_fn="v5",
    prompt_version="v4",
)

AGENT_RUN3_CONFIG = ExperimentConfig(
    name="run3-freeform",
    project="art-20q-runner-2026",
    reward_fn="v5",
    prompt_version="v6",
    question_mode="freeform",
)

AGENT_RUN4A_CONFIG = ExperimentConfig(
    name="run2-sft-v2",
    project="art-20q-runner-2026",
    reward_fn="v5",
    prompt_version="v4",
    perturbation_type="answer_corruption",
    perturbation_rate=0.15,
)

AGENT_RUN4B_CONFIG = ExperimentConfig(
    name="run2-sft-v2",
    project="art-20q-runner-2026",
    reward_fn="v5",
    prompt_version="v4",
    perturbation_type="forced_bad_start",
    perturbation_rate=0.0,  # not rate-based; always 2-3 forced questions
)

AGENT_RUN4C_CONFIG = ExperimentConfig(
    name="run2-sft-v2",
    project="art-20q-runner-2026",
    reward_fn="v5",
    prompt_version="v4",
    perturbation_type="attribute_removal",
    perturbation_rate=0.15,
)

AGENT_RUN5A_CONFIG = ExperimentConfig(
    name="run2-sft-v2",                # continue from SFT checkpoint
    project="art-20q-runner-2026",
    ppo=True,                          # PPO mode: clips to [0.8, 1.2]
)

AGENT_RUN5B_CONFIG = ExperimentConfig(
    name="run2-sft-v2",
    project="art-20q-runner-2026",
    epsilon=0.3,                       # tight GRPO clips: [0.7, 2.0]
    epsilon_high=1.0,
)

AGENT_RUN5C_CONFIG = ExperimentConfig(
    name="run2-sft-v2",
    project="art-20q-runner-2026",
    scale_rewards=False,               # no reward normalization (raw advantages)
)

AGENT_RUN5D_CONFIG = ExperimentConfig(
    name="run2-sft-v2",
    project="art-20q-runner-2026",
    beta=0.05,                         # KL penalty from SFT reference
)

CONFIGS: dict[str, ExperimentConfig] = {
    "001": AGENT_001_CONFIG,
    "002": AGENT_002_CONFIG,
    "002-v2": AGENT_002_V2_CONFIG,
    "002-v3": AGENT_002_V3_CONFIG,
    "004": AGENT_004_CONFIG,
    "005": AGENT_005_CONFIG,
    "005-v2": AGENT_005_V2_CONFIG,
    "006": AGENT_006_CONFIG,
    "run2": AGENT_RUN2_CONFIG,
    "run3": AGENT_RUN3_CONFIG,
    "run4a": AGENT_RUN4A_CONFIG,
    "run4b": AGENT_RUN4B_CONFIG,
    "run4c": AGENT_RUN4C_CONFIG,
    "run5a": AGENT_RUN5A_CONFIG,
    "run5b": AGENT_RUN5B_CONFIG,
    "run5c": AGENT_RUN5C_CONFIG,
    "run5d": AGENT_RUN5D_CONFIG,
}


def get_agent_config(version: str) -> ExperimentConfig:
    return CONFIGS.get(version, AGENT_006_CONFIG)
