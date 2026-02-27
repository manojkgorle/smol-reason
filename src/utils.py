"""Math parsing helpers and logging utilities.

These utilities handle the messy reality of parsing math answers from LLM outputs.
Math LLMs produce answers in many formats:
- Plain numbers: "42", "3.14", "-7"
- LaTeX: "\\frac{1}{2}", "\\sqrt{2}", "x^2 + 1"
- Boxed: "\\boxed{42}" (MATH dataset convention)
- Mixed: "The answer is $\\boxed{42}$."

The functions here extract and normalize these formats for reward computation.
"""

import re


def extract_think_content(text: str) -> str | None:
    """Extract content between <think> and </think> tags.

    Used to isolate the reasoning portion of model output for analysis.
    The <think> tags are the format we train the model to use for
    chain-of-thought reasoning.

    Returns None if no think tags are found (model hasn't learned the format yet).
    """
    # re.DOTALL makes . match newlines — reasoning content is often multi-line
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_answer_after_think(text: str) -> str:
    """Extract everything after the </think> tag (the final answer portion).

    In our format, the model should produce:
        <think>step-by-step reasoning</think>
        The answer is 42.

    This function extracts "The answer is 42." — the part that gets checked
    for correctness. The reasoning inside <think> is stripped because it
    might contain intermediate calculations that confuse answer extraction.
    """
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    # No think tags → treat entire text as the answer
    return text.strip()


def extract_boxed(text: str) -> str | None:
    r"""Extract content from \boxed{...}, handling nested braces.

    The MATH dataset uses \boxed{answer} to denote final answers.
    Models trained on math also learn to produce \boxed{} answers.

    Challenge: answers can have nested braces like \boxed{\frac{1}{2}}
    where \frac{}{} contains its own braces. We use brace-depth counting
    to find the matching closing brace.

    Examples:
        \boxed{42}           → "42"
        \boxed{\frac{1}{2}}  → "\frac{1}{2}"
        \boxed{\sqrt{x^{2}}} → "\sqrt{x^{2}}"
    """
    idx = text.find("\\boxed{")
    if idx == -1:
        return None
    depth = 0
    start = idx + len("\\boxed{")
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return text[start:i]
            depth -= 1
    return None


def extract_gsm8k_answer(solution: str) -> str:
    """Extract the numeric answer from a GSM8K solution.

    GSM8K answers follow the format:
        Step 1: calculation
        Step 2: calculation
        #### 42

    Everything after "####" is the ground truth answer.
    We also strip commas (e.g., "1,234" → "1234") for clean numeric parsing.
    """
    parts = solution.split("####")
    if len(parts) >= 2:
        return parts[-1].strip().replace(",", "")
    return solution.strip()


def has_self_correction(text: str) -> bool:
    """Detect self-correction patterns in reasoning text.

    Self-correction is one of the most interesting emergent behaviors from R1-Zero.
    The model learns to:
    - Question its own reasoning: "wait", "hmm"
    - Catch mistakes: "actually", "that's not right", "I made an error"
    - Redo work: "let me reconsider", "let me recheck"

    This is NOT taught — it emerges from RL because self-correcting leads to
    correct final answers, which leads to higher reward. The model essentially
    discovers that "checking your work" is a useful strategy.

    We track the self-correction rate during training as a signal of reasoning
    sophistication. Expect it to start near 0% and climb as training progresses.
    """
    patterns = [
        r"\bwait\b",
        r"\bactually\b",
        r"\blet me reconsider\b",
        r"\bI made (?:a |an )?(?:error|mistake)\b",
        r"\bthat(?:'s| is) (?:not right|wrong|incorrect)\b",
        r"\blet me (?:re)?check\b",
        r"\bno,\s",
        r"\bhmm\b",
    ]
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in patterns)


def count_think_tokens(text: str, tokenizer) -> int:
    """Count the number of tokens inside <think> tags.

    Useful for tracking reasoning length over training. In R1-Zero, the model's
    reasoning length grows substantially during training — it learns that longer,
    more detailed reasoning leads to correct answers.

    However, unbounded growth can also be a problem (verbose reasoning that
    wastes tokens). The max_completion_length in the config implicitly bounds this.
    """
    content = extract_think_content(text)
    if content is None:
        return 0
    return len(tokenizer.encode(content, add_special_tokens=False))
