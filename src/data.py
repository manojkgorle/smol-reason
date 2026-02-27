"""Dataset loading, answer extraction, and prompt formatting for GRPO training.

=== Data Pipeline Overview ===

We use two math datasets:

1. **GSM8K** (Grade School Math 8K):
   - 7,473 training / 1,319 test problems
   - Grade-school level word problems (arithmetic, basic algebra)
   - Answers are numbers after "####" delimiter
   - Example: "Natalia sold clips to 48 friends... #### 72"

2. **MATH** (Hendrycks' Competition Math):
   - 7,500 training / 5,000 test problems
   - Competition-level problems (algebra, geometry, number theory, etc.)
   - Answers in LaTeX \\boxed{} format
   - Example: "Find all solutions... \\boxed{\\frac{1}{2}}"

Combined: ~14,973 training prompts, covering a range from simple to hard.

=== Why These Datasets? ===

Math is ideal for RL training because:
- **Verifiable**: answers are objectively right or wrong (clear reward signal)
- **Requires reasoning**: multi-step problems benefit from chain-of-thought
- **Scalable difficulty**: GSM8K (easy) to MATH (hard) provides curriculum
- **Well-studied**: easy to compare results with published baselines

=== Dataset Format for GRPOTrainer ===

GRPOTrainer expects a dataset with a "prompt" column containing chat messages.
Any additional columns (like "solution") are automatically forwarded as keyword
arguments to reward functions. This is how the correctness reward gets access
to the ground truth answer.
"""

from datasets import Dataset, concatenate_datasets, load_dataset

from src.utils import extract_gsm8k_answer

# === System Prompt Design ===
#
# This is deliberately minimal. In the R1-Zero approach, we want RL to discover
# reasoning strategies, not prescribe them. The system prompt:
# - Hints at the <think> format (so the model knows the tags exist)
# - Does NOT show examples of reasoning
# - Does NOT specify how to format the final answer
#
# The RL process will learn:
# - WHAT to put inside <think> tags (reasoning steps)
# - HOW to format the final answer (so the reward function can parse it)
# - HOW MUCH reasoning to do (short for easy problems, long for hard ones)
#
# Compare with R1 (non-Zero): there, the model is first SFT'd on thousands of
# reasoning examples before RL. R1-Zero skips this entirely.
SYSTEM_PROMPT = (
    "You are a helpful assistant. When solving problems, think step by step "
    "inside <think>...</think> tags, then provide your final answer."
)


def load_gsm8k_train() -> Dataset:
    """Load GSM8K training set, formatted for GRPO.

    GSM8K format: each sample has "question" (the problem) and "answer"
    (multi-line solution ending with "#### <number>").

    We extract just the final number as the ground truth for reward computation.
    The full solution steps are discarded — we want RL to discover its own
    reasoning, not mimic the provided solutions.
    """
    ds = load_dataset("openai/gsm8k", "main", split="train")
    return ds.map(
        lambda ex: {
            "prompt": _make_prompt(ex["question"]),
            # Extract just the number after "####" — this is the ground truth
            # that gets passed to reward functions via **kwargs
            "solution": extract_gsm8k_answer(ex["answer"]),
        },
        remove_columns=ds.column_names,
    )


def load_gsm8k_test() -> Dataset:
    """Load GSM8K test set (1,319 problems) for evaluation."""
    ds = load_dataset("openai/gsm8k", "main", split="test")
    return ds.map(
        lambda ex: {
            "prompt": _make_prompt(ex["question"]),
            "solution": extract_gsm8k_answer(ex["answer"]),
        },
        remove_columns=ds.column_names,
    )


def load_math_train() -> Dataset:
    """Load MATH training set, formatted for GRPO.

    MATH format: each sample has "problem" and "solution" (containing \\boxed{answer}).
    We keep the full solution string because math_verify can parse \\boxed{} directly.
    """
    ds = load_dataset("hendrycks/competition_math", split="train")
    return ds.map(
        lambda ex: {
            "prompt": _make_prompt(ex["problem"]),
            # Keep full solution — math_verify will extract \\boxed{} for comparison
            "solution": ex["solution"],
        },
        remove_columns=ds.column_names,
    )


def load_math_test() -> Dataset:
    """Load MATH test set (5,000 problems) for evaluation."""
    ds = load_dataset("hendrycks/competition_math", split="test")
    return ds.map(
        lambda ex: {
            "prompt": _make_prompt(ex["problem"]),
            "solution": ex["solution"],
        },
        remove_columns=ds.column_names,
    )


def load_train_dataset() -> Dataset:
    """Load combined GSM8K + MATH training dataset, shuffled.

    Mixing datasets of different difficulty levels acts as a natural curriculum:
    - GSM8K problems provide easy wins early in training (reward signal)
    - MATH problems challenge the model as it improves
    - Shuffling ensures both difficulty levels appear in every training batch
    """
    gsm8k = load_gsm8k_train()
    math = load_math_train()
    combined = concatenate_datasets([gsm8k, math])
    return combined.shuffle(seed=42)


def load_test_datasets() -> dict[str, Dataset]:
    """Load test datasets for evaluation.

    Returns separate datasets so we can measure performance on each independently.
    This is important because GSM8K and MATH have very different difficulty levels —
    a model might do well on GSM8K but poorly on MATH.
    """
    return {
        "gsm8k": load_gsm8k_test(),
        "math": load_math_test(),
    }


def _make_prompt(question: str) -> list[dict[str, str]]:
    """Format a question into chat messages for GRPO.

    GRPOTrainer expects "prompt" to be a list of message dicts (chat format).
    It will apply the model's chat template to convert this to token IDs.

    For Qwen2.5, this becomes:
        <|im_start|>system
        You are a helpful assistant...
        <|im_end|>
        <|im_start|>user
        {question}
        <|im_end|>
        <|im_start|>assistant

    The model then generates the assistant's response.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


# === Canary Problems ===
#
# These are fixed, simple math problems used to track reasoning emergence.
# By evaluating the SAME problems at regular intervals during training,
# we can see how the model's behavior evolves:
#
# Step 0:   "12" (just the number, no reasoning)
# Step 100: "15% of 80... the answer is 12" (some text but no tags)
# Step 300: "<think>15% of 80 = 12</think>12" (tags appear! "aha moment")
# Step 500: "<think>To find 15% of 80, I multiply 0.15 * 80 = 12</think>
#            The answer is 12." (structured reasoning)
# Step 800: "<think>Let me calculate 15% of 80. 15/100 * 80 = 0.15 * 80 = 12.
#            Let me verify: 10% of 80 = 8, 5% of 80 = 4, 8+4 = 12. Correct.</think>
#            The answer is **12**." (self-verification emerges!)
CANARY_PROBLEMS = [
    "What is 15% of 80?",
    "If a train travels at 60 mph for 2.5 hours, how far does it travel?",
    "Solve for x: 3x + 7 = 22",
    "A store offers a 20% discount on a $45 item. What is the final price?",
    "What is the sum of the first 10 positive integers?",
]

CANARY_SOLUTIONS = ["12", "150", "5", "36", "55"]


def get_canary_prompts() -> list[list[dict[str, str]]]:
    """Return fixed canary prompts for tracking reasoning emergence."""
    return [_make_prompt(q) for q in CANARY_PROBLEMS]
