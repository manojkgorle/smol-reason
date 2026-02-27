"""Main GRPO training script for reasoning LLM.

Replicates the DeepSeek-R1-Zero experiment at 1.5B scale using pure RL (GRPO).
No SFT warm-up — reasoning emerges purely from reinforcement learning.

=== Background: What is GRPO? ===

Group Relative Policy Optimization (GRPO) is the RL algorithm used in DeepSeek-R1.
It's a variant of PPO (Proximal Policy Optimization) designed specifically for LLMs:

1. For each prompt, generate G completions (a "group")
2. Score each completion with reward functions
3. Compute "advantages" by comparing each completion's reward to the group mean
   (completions better than average get positive advantage, worse get negative)
4. Update the policy to increase probability of high-advantage completions
   and decrease probability of low-advantage ones, clipped to prevent
   too-large updates (controlled by epsilon)

Key difference from PPO: GRPO doesn't need a separate value/critic network.
The group statistics (mean, std) replace the value baseline, which saves memory
and simplifies training.

=== R1-Zero: Why Pure RL? ===

DeepSeek-R1-Zero showed that reasoning can emerge from RL alone, without any
supervised fine-tuning (SFT) on reasoning traces. The model learns to:
- Use <think> tags for chain-of-thought reasoning
- Self-correct ("wait, let me reconsider...")
- Show increasing reasoning depth over training

This is remarkable because the model was never shown examples of reasoning —
it discovered that thinking before answering leads to higher rewards.

Usage:
    python src/train_grpo.py --config configs/grpo_qwen2.5_1.5b.yaml
    python src/train_grpo.py --config configs/grpo_debug.yaml
"""

import torch
from transformers import AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser
from trl.rewards import reasoning_accuracy_reward

from src.data import get_canary_prompts, load_train_dataset, CANARY_SOLUTIONS
from src.rewards import format_reward
from src.utils import extract_think_content, has_self_correction


class EmergentBehaviorCallback(TrainerCallback):
    """Tracks reasoning emergence during training — the "aha moment" detector.

    === What is the "aha moment"? ===

    In the R1-Zero paper, there's a striking phase transition where the model
    suddenly starts using <think> tags and chain-of-thought reasoning, even though
    it was never explicitly taught to do so. This happens because:

    1. Early training: model gives short, often wrong answers → low reward
    2. By chance, some completions include reasoning-like text → slightly better
    3. RL amplifies this: reasoning → correct answers → reward → more reasoning
    4. Phase transition: model consistently uses structured reasoning

    This callback monitors for these signals every `eval_every` steps by:
    - Generating completions for the same fixed "canary" problems each time
    - Tracking % of completions using <think>...</think> tags
    - Measuring average reasoning length (tokens inside <think>)
    - Detecting self-correction patterns ("wait", "actually", "let me reconsider")

    Watching these metrics on W&B/TensorBoard reveals the emergence curve.
    """

    def __init__(self, tokenizer, eval_every: int = 50):
        self.tokenizer = tokenizer
        self.eval_every = eval_every
        self.canary_prompts = get_canary_prompts()

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_every != 0 or model is None:
            return

        model.eval()
        results = []

        for i, messages in enumerate(self.canary_prompts):
            # Apply the chat template (Qwen2.5 uses ChatML format):
            #   <|im_start|>system\n{system_prompt}<|im_end|>\n
            #   <|im_start|>user\n{question}<|im_end|>\n
            #   <|im_start|>assistant\n
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,  # Same temp as training for consistency
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Slice off the prompt tokens to get only the generated completion
            generated = outputs[0][inputs["input_ids"].shape[1]:]
            completion = self.tokenizer.decode(generated, skip_special_tokens=True)
            results.append(completion)

        # === Compute emergence metrics ===

        # Think usage: what % of canary completions use proper <think>...</think> tags?
        # Early in training this will be ~0%. Watch for the jump to >50%.
        think_usage = sum(1 for r in results if "<think>" in r and "</think>" in r) / len(results)
        think_lengths = []
        self_corrections = 0

        for r in results:
            content = extract_think_content(r)
            if content:
                think_lengths.append(len(self.tokenizer.encode(content, add_special_tokens=False)))
                # Self-correction: a key emergent behavior from R1-Zero.
                # The model learns to question its own reasoning mid-stream.
                if has_self_correction(content):
                    self_corrections += 1

        avg_think_len = sum(think_lengths) / len(think_lengths) if think_lengths else 0
        self_correction_rate = self_corrections / len(results)

        logs = {
            "emergence/think_usage_pct": think_usage * 100,
            "emergence/avg_think_tokens": avg_think_len,
            "emergence/self_correction_rate": self_correction_rate,
        }

        # Log full canary completions so you can read them on W&B
        # and qualitatively see how reasoning evolves
        for i, (r, sol) in enumerate(zip(results, CANARY_SOLUTIONS)):
            logs[f"canary/problem_{i}"] = r[:500]

        if state.log_history is not None:
            state.log_history.append(logs)

        if hasattr(self, "_trainer"):
            self._trainer.log(logs)

        model.train()

    def set_trainer(self, trainer):
        self._trainer = trainer


def main():
    # === Step 1: Parse config ===
    # TrlParser is TRL's config parser. It reads a YAML config file (--config flag)
    # and maps fields to the appropriate dataclass:
    #   - GRPOConfig: all training hyperparameters (lr, batch size, GRPO-specific params)
    #   - ModelConfig: model name, dtype, trust_remote_code
    # CLI arguments override YAML values, so you can do:
    #   python train_grpo.py --config base.yaml --max_steps 500
    parser = TrlParser((GRPOConfig, ModelConfig))
    grpo_config, model_config = parser.parse_args_and_config()

    # === Step 2: Load tokenizer ===
    # We load the tokenizer separately (even though GRPOTrainer can auto-load it)
    # because we need it for the EmergentBehaviorCallback to generate and decode text.
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
    )
    # Qwen2.5-1.5B base doesn't have a pad token set by default.
    # We reuse eos_token — this is standard practice for decoder-only models.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # === Step 3: Load dataset ===
    # Combined GSM8K (7,473) + MATH (7,500) = ~14,973 training prompts.
    # Each sample has: {"prompt": [system_msg, user_msg], "solution": "ground_truth"}
    # The "solution" column is automatically forwarded as a kwarg to reward functions.
    print("Loading training dataset (GSM8K + MATH)...")
    train_dataset = load_train_dataset()
    print(f"Training on {len(train_dataset)} prompts")

    # === Step 4: Define reward functions ===
    # GRPO needs reward functions to score completions. We use two:
    #
    # 1. reasoning_accuracy_reward (from TRL, weight=1.0):
    #    - Strips everything inside <think>...</think> tags
    #    - Extracts the final answer from the remaining text
    #    - Uses math_verify library for robust comparison (handles LaTeX, fractions,
    #      symbolic equivalence like "1/2" == "0.5" == "\frac{1}{2}")
    #    - Returns 1.0 (correct), 0.0 (wrong), or None (unparseable gold → skip)
    #
    # 2. format_reward (custom, weight=0.2):
    #    - Checks for proper <think>content</think>answer structure
    #    - Graded: 0.0 → 0.25 → 0.5 → 1.0 (provides gradient signal for
    #      incremental format adoption, unlike binary 0/1)
    #
    # The weights [1.0, 0.2] mean correctness dominates. This prevents reward
    # hacking where the model learns to produce perfect format but wrong answers.
    # The format reward provides a gentle nudge toward structured reasoning.
    reward_funcs = [reasoning_accuracy_reward, format_reward]

    # === Step 5: Set up emergence tracking ===
    emergence_cb = EmergentBehaviorCallback(tokenizer, eval_every=50)

    # === Step 6: Initialize GRPOTrainer ===
    # Passing model as a string (not a loaded model) tells GRPOTrainer to handle
    # loading internally. It uses model_init_kwargs from the config for parameters
    # like torch_dtype and trust_remote_code.
    #
    # What happens inside GRPOTrainer on each training step:
    #   1. Sample a batch of prompts from train_dataset
    #   2. Generate G completions per prompt (G = num_generations)
    #   3. Score all completions with reward_funcs
    #   4. Compute advantages: A_i = (reward_i - mean(rewards)) / std(rewards)
    #      (when scale_rewards=True; we use False per Dr. GRPO recommendation)
    #   5. Compute policy ratio: r = pi_new(completion) / pi_old(completion)
    #   6. GRPO loss = -min(r * A, clip(r, 1-eps, 1+eps) * A)
    #      (clipped surrogate objective, same as PPO)
    #   7. Backprop and update weights
    trainer = GRPOTrainer(
        model=model_config.model_name_or_path,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[emergence_cb],
    )

    emergence_cb.set_trainer(trainer)

    # === Step 7: Train ===
    # resume_from_checkpoint enables Colab resilience — if the session disconnects,
    # you can restart from the last checkpoint saved to Google Drive.
    trainer.train(resume_from_checkpoint=grpo_config.resume_from_checkpoint)

    # === Step 8: Save ===
    trainer.save_model(grpo_config.output_dir)
    tokenizer.save_pretrained(grpo_config.output_dir)
    print(f"Model saved to {grpo_config.output_dir}")


if __name__ == "__main__":
    main()
