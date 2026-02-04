import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from environment import TwentyQuestionsEpisode

def check_episode_finished(ep: 'TwentyQuestionsEpisode') -> bool:
    """Duplicate this helper here to avoid circular import"""
    if ep["done"]:
        return True
    if ep["questions_asked"] >= 15:  # MAX_QUESTIONS
        return True
    return False

def compute_reward_v1(ep: 'TwentyQuestionsEpisode') -> float:
    """Original reward function - led to reward hacking"""
    UNKNOWN_ATTR_PENALTY = 0.20
    QUESTION_COST = 0.05
    r = 0.0
    r -= QUESTION_COST * ep["questions_asked"]
    r -= UNKNOWN_ATTR_PENALTY * ep["invalid_questions"]

    if not check_episode_finished(ep):
        return r
    
    if ep["guessed_id"] is None:
        r -= 1.0
    else:
        r += 1.0 if (ep["guessed_id"] == ep["secret_id"]) else -1.0
    return r

def compute_reward_v2(ep: 'TwentyQuestionsEpisode') -> float:
    """Improved reward - incentivizes asking questions and narrowing candidates"""
    QUESTION_COST = 0.02
    UNKNOWN_ATTR_PENALTY = 0.20
    CORRECT_REWARD = 2.0
    WRONG_REWARD = -2.0
    TIMEOUT_REWARD = -2.0
    N0 = 76
    NARROW_BONUS = 0.6

    r = 0.0
    r -= QUESTION_COST * ep["questions_asked"]
    r -= UNKNOWN_ATTR_PENALTY * ep["invalid_questions"]

    N = max(1, len(ep["candidates"]))
    r += NARROW_BONUS * (1.0 - (math.log(N) / math.log(N0)))

    if not check_episode_finished(ep):
        return r

    if ep["guessed_id"] is None:
        r += TIMEOUT_REWARD
    else:
        r += CORRECT_REWARD if (ep["guessed_id"] == ep["secret_id"]) else WRONG_REWARD

    return r

def compute_reward_v3(ep: 'TwentyQuestionsEpisode') -> float:
    """v3: Stronger incentives for narrowing + correct guessing"""
    QUESTION_COST = 0.01  # Reduced from 0.02 - make questions cheaper
    UNKNOWN_ATTR_PENALTY = 0.30  # Increased from 0.20 - punish invalid questions harder
    CORRECT_REWARD = 5.0  # Increased from 2.0 - BIG reward for correct guess
    WRONG_REWARD = -3.0  # Increased penalty from -2.0
    TIMEOUT_REWARD = -3.0  # Increased penalty from -2.0
    N0 = 76
    NARROW_BONUS = 1.0  # Increased from 0.6 - reward narrowing more

    r = 0.0
    r -= QUESTION_COST * ep["questions_asked"]
    r -= UNKNOWN_ATTR_PENALTY * ep["invalid_questions"]

    # Reward narrowing (logarithmic)
    N = max(1, len(ep["candidates"]))
    r += NARROW_BONUS * (1.0 - (math.log(N) / math.log(N0)))

    if not check_episode_finished(ep):
        return r

    if ep["guessed_id"] is None:
        r += TIMEOUT_REWARD
    else:
        r += CORRECT_REWARD if (ep["guessed_id"] == ep["secret_id"]) else WRONG_REWARD

    return r

def compute_reward_v4(ep: 'TwentyQuestionsEpisode') -> float:
    """v4: Force strategic questioning AND guessing"""
    QUESTION_COST = 0.05  # Increased from 0.01 - make questions more expensive
    UNKNOWN_ATTR_PENALTY = 0.50  # Increased - punish invalid questions harder
    CORRECT_REWARD = 10.0  # Massive reward for winning
    WRONG_REWARD = -5.0   # Big penalty for wrong guess
    TIMEOUT_PENALTY = -8.0  # HUGE penalty for not guessing - force a guess
    N0 = 76
    NARROW_BONUS = 0.8  # Slightly reduced
    
    r = 0.0
    r -= QUESTION_COST * ep["questions_asked"]
    r -= UNKNOWN_ATTR_PENALTY * ep["invalid_questions"]
    
    # Reward narrowing
    N = max(1, len(ep["candidates"]))
    r += NARROW_BONUS * (1.0 - (math.log(N) / math.log(N0)))
    
    if not check_episode_finished(ep):
        return r
    
    if ep["guessed_id"] is None:
        r += TIMEOUT_PENALTY  # Much worse than wrong guess!
    else:
        r += CORRECT_REWARD if (ep["guessed_id"] == ep["secret_id"]) else WRONG_REWARD
    
    return r

def compute_reward(ep: 'TwentyQuestionsEpisode') -> float:
    """Dispatch to correct reward function"""
    if ep.get('reward_fn') == 'v1':
        return compute_reward_v1(ep)
    elif ep.get('reward_fn') == 'v3':
        return compute_reward_v3(ep)
    elif ep.get('reward_fn') == 'v4':
        return compute_reward_v4(ep)
    return compute_reward_v2(ep)  # default to v2