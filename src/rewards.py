"""Reward functions for GRPO training.

=== Why Rewards Matter So Much in RL for LLMs ===

In GRPO, the reward signal is the ONLY teaching signal. Unlike SFT where you show
the model "here's a good response, learn to produce this", in RL you say "here's
how good your response was, figure out how to do better."

This means:
- A bad reward function → the model learns the wrong thing (reward hacking)
- A sparse reward (only 0 or 1) → slow learning, hard to improve incrementally
- A well-shaped reward → smooth gradient signal, faster and more stable learning

=== Our Reward Design ===

We use two reward functions with different weights:

1. **Correctness reward** (weight=1.0): TRL's built-in `reasoning_accuracy_reward`
   - This is the primary signal: did the model get the right answer?
   - Uses `math_verify` for robust math answer comparison
   - Handles LaTeX, fractions, symbolic equivalence

2. **Format reward** (weight=0.2): Our custom graded reward
   - This is a secondary shaping signal: is the model using <think> tags?
   - Graded (0.0 → 0.25 → 0.5 → 1.0) rather than binary (0/1)
   - The grading provides denser gradient signal for incremental format adoption

The weight ratio (1.0 vs 0.2) is important:
- Correctness dominates, so the model can't hack the format reward by producing
  perfect <think> tags with wrong answers
- Format reward is just enough to nudge the model toward structured reasoning
- As the model improves at reasoning, correct answers naturally follow

=== Why Not Just Use Correctness Alone? ===

You could! The R1-Zero paper used only correctness + a format reward.
The format reward helps because:
- It provides learning signal even when the answer is wrong
  (a wrong answer with <think> tags scores 0.0 + 0.2 = 0.2, vs 0.0 + 0.0 = 0.0)
- It creates a gentle gradient toward the structured format
- Once the model uses the format, the reasoning inside <think> naturally improves
"""

import re


def format_reward(completions: list, **kwargs) -> list[float]:
    """Graded format compliance reward for <think>...</think> tags.

    === Reward Function Signature (TRL Convention) ===

    All reward functions for GRPOTrainer must follow this signature:
        def reward_fn(completions: list, **kwargs) -> list[float]

    - completions: list of completions, where each completion is a list of message
      dicts like [{"role": "assistant", "content": "..."}]
    - **kwargs: any extra dataset columns (like "solution") are auto-forwarded
    - Returns: list of float rewards, one per completion

    === Grading Scale ===

    Instead of binary 0/1 (like TRL's built-in think_format_reward), we use a
    graded scale. This provides denser gradient signal:

    - 1.0:  proper <think>reasoning</think>answer — the ideal format
    - 0.5:  has both tags but incomplete (empty think, or no answer after)
    - 0.25: only one tag present (partial format adoption)
    - 0.0:  no tags at all

    The graded reward helps the model learn incrementally:
    - First it learns that <think> tags exist (0.0 → 0.25)
    - Then it learns to close them (0.25 → 0.5)
    - Then it learns to put reasoning inside and answer after (0.5 → 1.0)
    """
    rewards = []
    for completion in completions:
        # GRPOTrainer passes completions as [{"role": "assistant", "content": "..."}]
        text = completion[0]["content"] if isinstance(completion, list) else completion
        has_open = "<think>" in text
        has_close = "</think>" in text

        if not has_open and not has_close:
            # No format adoption at all
            rewards.append(0.0)
            continue

        if not has_open or not has_close:
            # Partial: model is learning the format exists but hasn't mastered it
            rewards.append(0.25)
            continue

        # Both tags present — now check the content structure
        match = re.search(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
        if match:
            think_content = match.group(1).strip()
            after_content = match.group(2).strip()
            if think_content and after_content:
                # Perfect: reasoning inside tags + answer outside
                rewards.append(1.0)
            elif think_content or after_content:
                # One part missing (e.g., empty think but has answer, or reasoning but no answer)
                rewards.append(0.5)
            else:
                # Both tags present but both empty
                rewards.append(0.25)
        else:
            # Tags present but regex didn't match (unusual edge case)
            rewards.append(0.25)

    return rewards
