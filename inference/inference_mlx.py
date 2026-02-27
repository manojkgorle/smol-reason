"""MLX inference on MacBook for the trained reasoning model.

=== Local Inference Pipeline ===

After training on a GPU server (Colab A100), we want to run the model locally
on a MacBook for interactive use. The pipeline is:

1. Train on GPU → save HuggingFace checkpoint
2. Convert to MLX format (convert_model.py) → optionally quantize to 4-bit
3. Load and run with mlx-lm (this script) → fast local inference

=== Why MLX instead of PyTorch on Mac? ===

While PyTorch works on Mac (via MPS backend), MLX is purpose-built for Apple Silicon:
- Better memory management (unified memory architecture)
- Faster token generation for autoregressive models
- Native quantization support
- ~2-3x faster than PyTorch MPS for inference

=== Display Format ===

When the model outputs <think>reasoning</think>answer, we display them separately:
- "Reasoning" section: shows the chain-of-thought (the interesting part!)
- "Answer" section: shows the final answer

This lets you see HOW the model reasons, not just what it answers.

Usage:
    python inference/inference_mlx.py --model_path outputs/grpo-qwen2.5-1.5b-mlx
    python inference/inference_mlx.py --model_path outputs/grpo-qwen2.5-1.5b-mlx --interactive
    python inference/inference_mlx.py --model_path outputs/grpo-qwen2.5-1.5b-mlx --batch problems.txt
"""

import argparse
import time

from mlx_lm import generate, load

# Same system prompt as training — consistency is important!
# If you change this, the model may behave differently than during training.
SYSTEM_PROMPT = (
    "You are a helpful assistant. When solving problems, think step by step "
    "inside <think>...</think> tags, then provide your final answer."
)


def format_prompt(question: str, tokenizer) -> str:
    """Format question with system prompt using the model's chat template.

    Uses the tokenizer's built-in chat template to ensure the prompt format
    exactly matches what the model saw during training. For Qwen2.5, this
    produces ChatML format with <|im_start|> and <|im_end|> tokens.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def display_response(text: str):
    """Display response with reasoning and answer separated.

    This makes it easy to see the model's chain-of-thought reasoning
    (the key output of R1-Zero-style training) separately from the answer.
    """
    if "<think>" in text and "</think>" in text:
        parts = text.split("</think>", 1)
        think_part = parts[0].replace("<think>", "").strip()
        answer_part = parts[1].strip() if len(parts) > 1 else ""

        print("\n--- Reasoning ---")
        print(think_part)
        print("\n--- Answer ---")
        print(answer_part)
    else:
        # Model didn't use think tags (early training or non-math prompt)
        print("\n--- Response ---")
        print(text)


def interactive_mode(model, tokenizer, max_tokens: int):
    """Run interactive inference loop.

    Type math problems and see the model reason through them in real-time.
    The generation uses temp=0.7 (same as training) for diverse reasoning.
    """
    print("Interactive mode. Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        prompt = format_prompt(question, tokenizer)

        start = time.time()
        response = generate(
            model, tokenizer, prompt=prompt,
            max_tokens=max_tokens, temp=0.7,
        )
        elapsed = time.time() - start

        display_response(response)

        # Show generation speed — useful for benchmarking
        tokens = len(tokenizer.encode(response))
        print(f"\n[{tokens} tokens, {elapsed:.1f}s, {tokens/elapsed:.0f} tok/s]\n")


def batch_mode(model, tokenizer, input_file: str, max_tokens: int):
    """Run batch inference on a file of questions (one per line).

    Uses greedy decoding (temp=0) for consistent, reproducible results.
    """
    with open(input_file) as f:
        questions = [line.strip() for line in f if line.strip()]

    print(f"Processing {len(questions)} questions...\n")

    for i, question in enumerate(questions):
        print(f"=== Problem {i+1}/{len(questions)} ===")
        print(f"Q: {question}")

        prompt = format_prompt(question, tokenizer)
        response = generate(
            model, tokenizer, prompt=prompt,
            max_tokens=max_tokens, temp=0.0,  # Greedy for batch consistency
        )
        display_response(response)
        print()


def main():
    parser = argparse.ArgumentParser(description="MLX inference for reasoning model")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--batch", type=str, default=None, help="Batch file (one question per line)")
    parser.add_argument("--question", type=str, default=None, help="Single question")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load(args.model_path)
    print("Model loaded.\n")

    if args.interactive:
        interactive_mode(model, tokenizer, args.max_tokens)
    elif args.batch:
        batch_mode(model, tokenizer, args.batch, args.max_tokens)
    elif args.question:
        prompt = format_prompt(args.question, tokenizer)
        response = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens, temp=0.0)
        display_response(response)
    else:
        interactive_mode(model, tokenizer, args.max_tokens)


if __name__ == "__main__":
    main()
