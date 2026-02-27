"""Evaluation script for trained reasoning model.

=== Evaluation Strategy ===

We measure four key metrics:

1. **Pass@1 accuracy** (greedy, temp=0):
   The most important metric — can the model get the right answer?
   We use greedy decoding (temp=0) for deterministic evaluation.
   - GSM8K baseline (Qwen2.5-1.5B raw): ~40-50%
   - After GRPO training (1000 steps): expect ~55-65%
   - For reference, DeepSeek-R1-Zero-Qwen-32B reached ~70%+ on GSM8K

2. **Format compliance rate**:
   What % of responses use proper <think>...</think> tags?
   This tells us whether the model learned the structured reasoning format.
   - Before training: ~0% (base model doesn't know about <think> tags)
   - After training: should reach >90%

3. **Average reasoning length** (tokens inside <think>):
   How much does the model "think" before answering?
   - Before training: 0 (no tags)
   - After training: typically 100-500 tokens depending on problem difficulty
   - Longer reasoning generally correlates with harder problems

4. **Self-correction rate**:
   How often does the model catch and fix its own mistakes?
   This is the most "human-like" reasoning behavior.
   - Before training: 0%
   - After training: typically 5-15% of responses

=== Why Separate from Training Eval? ===

We run evaluation as a standalone script rather than using GRPOTrainer's built-in
eval because:
- We want greedy decoding (temp=0) — training uses temp=0.7
- We want detailed per-sample results for analysis
- We can eval on the full test set without slowing training
- We can compare multiple checkpoints side-by-side

Usage:
    python src/evaluate.py --model_path outputs/grpo-qwen2.5-1.5b --num_samples 200
    python src/evaluate.py --model_path Qwen/Qwen2.5-1.5B  # baseline
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import load_test_datasets
from src.utils import (
    extract_answer_after_think,
    extract_boxed,
    extract_think_content,
    has_self_correction,
)


def check_answer_correct(prediction: str, gold: str) -> bool:
    """Check if model's prediction matches the gold answer.

    This is a simplified correctness checker for standalone evaluation.
    During training, we use TRL's reasoning_accuracy_reward which uses
    the full math_verify library for more robust comparison.

    The checking cascade:
    1. Strip <think> tags to get just the answer portion
    2. Extract \\boxed{} content if present (for MATH-style answers)
    3. Try numeric comparison (handles "42" == "42.0" == "42.00")
    4. Fall back to case-insensitive string comparison
    """
    # Step 1: Get just the answer part (after </think> if present)
    pred_answer = extract_answer_after_think(prediction)

    # Step 2: Extract boxed content if present
    # Both prediction and gold might use \boxed{}
    pred_boxed = extract_boxed(pred_answer)
    gold_boxed = extract_boxed(gold)

    pred_str = (pred_boxed or pred_answer).strip()
    gold_str = (gold_boxed or gold).strip()

    # Step 3: Clean up common formatting differences
    pred_clean = pred_str.replace(",", "").replace("$", "").replace("%", "").strip()
    gold_clean = gold_str.replace(",", "").replace("$", "").replace("%", "").strip()

    # Step 4: Try numeric comparison first (most GSM8K answers are numbers)
    try:
        pred_num = float(pred_clean)
        gold_num = float(gold_clean)
        return abs(pred_num - gold_num) < 1e-6
    except ValueError:
        pass

    # Step 5: Fall back to string comparison (for MATH symbolic answers)
    return pred_clean.lower() == gold_clean.lower()


def evaluate_model(
    model,
    tokenizer,
    dataset,
    num_samples: int | None = None,
    max_new_tokens: int = 2048,
    batch_size: int = 1,
) -> dict:
    """Evaluate model on a dataset, computing all metrics.

    Uses greedy decoding (temperature=0, do_sample=False) for reproducible results.
    This is different from training which uses temperature=0.7 with sampling.
    """
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    correct = 0
    total = 0
    format_ok = 0
    think_lengths = []
    self_corrections = 0
    results = []

    model.eval()
    for i in tqdm(range(len(dataset)), desc="Evaluating"):
        sample = dataset[i]
        messages = sample["prompt"]
        gold = sample["solution"]

        # Apply chat template to convert messages → token string
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                # Greedy decoding for eval: deterministic, reproducible
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Slice off prompt tokens to get only the generated part
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        completion = tokenizer.decode(generated, skip_special_tokens=True)

        # Metric 1: Correctness
        is_correct = check_answer_correct(completion, gold)
        if is_correct:
            correct += 1
        total += 1

        # Metric 2: Format compliance
        has_think = "<think>" in completion and "</think>" in completion
        if has_think:
            format_ok += 1

        # Metrics 3 & 4: Reasoning length and self-correction
        think_content = extract_think_content(completion)
        if think_content:
            think_lengths.append(len(tokenizer.encode(think_content, add_special_tokens=False)))
            if has_self_correction(think_content):
                self_corrections += 1

        # Save detailed results for qualitative analysis
        results.append({
            "question": messages[-1]["content"][:200],
            "gold": gold,
            "prediction": completion[:500],
            "correct": is_correct,
            "has_think": has_think,
        })

    metrics = {
        "accuracy": correct / total if total > 0 else 0,
        "total": total,
        "correct": correct,
        "format_compliance": format_ok / total if total > 0 else 0,
        "avg_think_tokens": sum(think_lengths) / len(think_lengths) if think_lengths else 0,
        "self_correction_rate": self_corrections / total if total > 0 else 0,
    }

    return metrics, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate reasoning model")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=None, help="Limit samples per dataset")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--datasets", nargs="+", default=["gsm8k", "math"])
    args = parser.parse_args()

    # Load model with automatic dtype selection:
    # - bfloat16 on GPU (faster, lower memory)
    # - float32 on CPU (bfloat16 not well-supported on CPU)
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",  # Automatically places layers on available devices
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading test datasets...")
    test_datasets = load_test_datasets()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    for name in args.datasets:
        if name not in test_datasets:
            print(f"Skipping unknown dataset: {name}")
            continue

        print(f"\nEvaluating on {name}...")
        metrics, results = evaluate_model(
            model, tokenizer, test_datasets[name],
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
        )

        all_metrics[name] = metrics
        print(f"  Accuracy: {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['total']})")
        print(f"  Format compliance: {metrics['format_compliance']:.1%}")
        print(f"  Avg think tokens: {metrics['avg_think_tokens']:.0f}")
        print(f"  Self-correction rate: {metrics['self_correction_rate']:.1%}")

        # Save per-sample results (first 50) for qualitative analysis
        with open(output_dir / f"{name}_results.json", "w") as f:
            json.dump({"metrics": metrics, "samples": results[:50]}, f, indent=2)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
