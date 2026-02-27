# reson-llm: Teaching an LLM to Reason with Pure RL

Train a reasoning LLM from scratch using Group Relative Policy Optimization (GRPO) — replicating the [DeepSeek-R1-Zero](https://arxiv.org/abs/2501.12948) experiment at 1.5B scale. No supervised reasoning data. The model discovers chain-of-thought reasoning entirely through reinforcement learning.

## What This Project Does

Starting from a base language model (Qwen2.5-1.5B) that has **no reasoning training**, we apply GRPO with math problem rewards. Over ~1000 training steps, the model spontaneously learns to:

1. Use `<think>...</think>` tags for chain-of-thought reasoning
2. Break problems into steps
3. Self-correct mistakes ("wait, that's not right...")
4. Produce structured final answers

This is the "aha moment" from the R1-Zero paper — reasoning emerges from RL alone.

## Background: How Does This Work?

### The Core Idea

Traditional LLM training for reasoning uses supervised fine-tuning (SFT): you show the model examples of good reasoning and it learns to imitate them. **R1-Zero takes a different approach**: give the model a math problem, let it generate an answer, tell it if it's right or wrong, and let it figure out the rest.

### GRPO Algorithm

[GRPO](https://arxiv.org/abs/2402.03300) (Group Relative Policy Optimization) is a variant of PPO designed for LLMs:

```
For each training step:
  1. Sample a batch of math problems (prompts)
  2. For each prompt, generate G=16 different completions
  3. Score each completion with reward functions:
     - Did it get the right answer? (0 or 1)
     - Did it use <think> tags? (0 to 1, graded)
  4. Compute advantages within each group:
     advantage = completion_reward - mean(group_rewards)
     (completions better than average get positive advantage)
  5. Update the model to make high-advantage completions more likely
     and low-advantage completions less likely
  6. Clip updates to prevent too-large changes (epsilon=0.2)
```

Key insight: **no value network needed**. Unlike PPO which trains a separate critic to estimate how good a state is, GRPO uses the group statistics (mean reward) as the baseline. This saves memory and simplifies training.

### Why Reasoning Emerges

The reward is only given for **correct final answers**. But to get correct answers on multi-step math problems, the model needs to reason. So the RL process discovers that:

1. Thinking before answering → more correct answers → higher reward
2. Structured thinking (`<think>` tags) → more organized reasoning → even better
3. Checking work ("wait, let me verify") → catches errors → higher reward

None of this is explicitly taught. The model discovers it through trial and error, guided only by the binary correctness signal.

## Project Structure

```
reson-llm/
├── pyproject.toml                     # Dependencies + project config
├── requirements.txt                   # Pinned deps for Colab
├── requirements-mac.txt               # MacBook inference deps (mlx-lm)
├── configs/
│   ├── grpo_qwen2.5_1.5b.yaml        # Full training config (A100, 1000 steps)
│   └── grpo_debug.yaml               # Debug config (CPU, 50 steps)
├── src/
│   ├── rewards.py                     # Reward functions (correctness + format)
│   ├── data.py                        # Dataset loading + prompt formatting
│   ├── train_grpo.py                  # Main training script
│   ├── evaluate.py                    # Evaluation on GSM8K/MATH test sets
│   └── utils.py                       # Math parsing helpers
├── inference/
│   ├── convert_model.py               # HuggingFace → MLX conversion
│   └── inference_mlx.py               # MacBook inference (interactive/batch)
├── notebooks/
│   └── train_grpo_colab.ipynb         # Self-contained Colab notebook
└── tests/
    ├── test_rewards.py                # Reward function tests
    └── test_data.py                   # Data pipeline tests
```

### Key Files Explained

| File | Purpose | Read This If... |
|------|---------|-----------------|
| `src/train_grpo.py` | Orchestrates the full training pipeline | You want to understand the training loop |
| `src/rewards.py` | Defines how completions are scored | You want to understand the reward signal |
| `src/data.py` | Loads and formats math datasets | You want to understand the training data |
| `configs/grpo_qwen2.5_1.5b.yaml` | All hyperparameters with explanations | You want to understand the config choices |

Every file is extensively commented explaining **why** each decision was made.

## Setup

### For Training (GPU)

```bash
pip install -r requirements.txt
```

Key dependencies:
- `trl>=0.15` — HuggingFace's RL training library (provides GRPOTrainer)
- `math-verify>=0.7` — Robust math answer verification (LaTeX, fractions, sympy)
- `wandb` — Experiment tracking

### For MacBook Inference

```bash
pip install -r requirements-mac.txt
```

## Quick Start

### 1. Verify the Pipeline (Local, ~5 min)

```bash
# Run tests first
pytest tests/ -v

# Debug run — verifies everything works (downloads model on first run)
python src/train_grpo.py --config configs/grpo_debug.yaml
```

### 2. Train on Colab (A100, ~5-8h)

Open `notebooks/train_grpo_colab.ipynb` in Google Colab:
- Select A100 GPU runtime
- The notebook auto-detects GPU memory (40GB vs 80GB) and adjusts config
- Checkpoints save to Google Drive every 100 steps
- Resume from checkpoint if session disconnects

Or train from the command line on any GPU server:

```bash
python src/train_grpo.py --config configs/grpo_qwen2.5_1.5b.yaml
```

### 3. Evaluate

```bash
# Evaluate trained model
python src/evaluate.py --model_path outputs/grpo-qwen2.5-1.5b --num_samples 200

# Evaluate baseline (raw Qwen2.5-1.5B) for comparison
python src/evaluate.py --model_path Qwen/Qwen2.5-1.5B --num_samples 200
```

### 4. Run on MacBook

```bash
# Convert to MLX format (optional: add --quantize 4bit)
python inference/convert_model.py --model_path outputs/grpo-qwen2.5-1.5b

# Interactive inference
python inference/inference_mlx.py --model_path outputs/grpo-qwen2.5-1.5b-mlx --interactive
```

## Training Config Deep Dive

The config file (`configs/grpo_qwen2.5_1.5b.yaml`) is extensively commented, but here's a summary of the most important hyperparameters:

### GRPO-Specific

| Parameter | Value | Why |
|-----------|-------|-----|
| `num_generations` | 16 | Group size for advantage estimation. R1-Zero used 16. Larger = more stable but more memory. |
| `beta` | 0.0 | No KL penalty. Saves memory (no reference model). DAPO paper showed this works with proper clipping. |
| `epsilon` | 0.2 | PPO clip range. Prevents >20% policy change per step. Standard, stable value. |
| `loss_type` | `dr_grpo` | Removes response-length bias. Without this, the model learns to give short answers. |
| `scale_rewards` | `false` | Avoids question-difficulty bias. Dr. GRPO recommendation. |

### Reward Functions

| Function | Weight | Source | What It Does |
|----------|--------|--------|-------------|
| `reasoning_accuracy_reward` | 1.0 | TRL built-in | Strips `<think>` tags, checks answer vs ground truth using `math_verify` |
| `format_reward` | 0.2 | Custom | Graded check for `<think>content</think>answer` structure (0.0→0.25→0.5→1.0) |

### Training

| Parameter | Value | Why |
|-----------|-------|-----|
| `learning_rate` | 1e-6 | 10x lower than SFT. RL gradients are noisier, needs conservative updates. |
| `max_grad_norm` | 0.2 | Tight gradient clipping for RL stability. |
| `max_steps` | 1000 | ~1 pass through the data. Extend to 2000+ for better results. |
| `gradient_checkpointing` | true | Trades compute for memory. Essential for G=16 on A100. |

## What to Watch During Training

Monitor these metrics on W&B/TensorBoard:

### Reward Curves
- **`reward/mean`**: should increase over training (model getting more answers right)
- **`reward/std`**: high early (diverse quality), decreases as model improves

### Emergence Metrics (logged every 50 steps)
- **`emergence/think_usage_pct`**: % of canary completions using `<think>` tags. Watch for the jump from ~0% to >50% — this is the "aha moment".
- **`emergence/avg_think_tokens`**: average reasoning length. Expect growth from 0 to 100-500 tokens.
- **`emergence/self_correction_rate`**: how often the model corrects itself. A sign of sophisticated reasoning.

### Training Stability
- **`loss`**: should decrease, but RL loss is noisy — look at the trend, not individual steps
- **`grad_norm`**: should stay below `max_grad_norm` (0.2). Spikes indicate instability.

## Expected Results

Results will vary, but approximate expectations for 1000 steps on A100:

| Metric | Baseline (Qwen2.5-1.5B) | After GRPO |
|--------|-------------------------|------------|
| GSM8K accuracy | ~40-50% | ~55-65% |
| MATH accuracy | ~15-25% | ~20-35% |
| Format compliance | ~0% | >90% |
| Self-correction rate | 0% | ~5-15% |

The biggest improvement is on GSM8K (grade-school math) because these problems are most amenable to the step-by-step reasoning the model learns. MATH competition problems are harder and may need more training steps or a larger model.

## Troubleshooting

### Out of Memory on GPU
- Reduce `num_generations` (16 → 8 → 4)
- Reduce `max_completion_length` (2048 → 1024 → 512)
- Reduce `per_device_train_batch_size` (2 → 1)

### Training Reward Not Increasing
- Check that reward functions are working: `pytest tests/test_rewards.py -v`
- Increase `num_generations` for better advantage estimation
- Try `beta: 0.01` to add a small KL penalty (prevents too-fast divergence)

### Model Outputs Gibberish After Training
- Learning rate too high — try 5e-7
- Training diverged — check W&B for reward/loss spikes
- Resume from an earlier checkpoint

### Colab Disconnects
- Checkpoints save to Google Drive every 100 steps
- The notebook auto-detects and resumes from the latest checkpoint
- Use the Colab "keep-alive" trick (see notebook)

## Architecture Decisions

### Why Qwen2.5-1.5B?

- **Base model** (not Instruct): clean slate for RL to work with
- **1.5B params**: smallest size where reasoning emergence is plausible. Fits on A100 with G=16.
- **Qwen2.5 family**: strong base capabilities, good tokenizer, widely used in RL research
- **Fallback**: if 1.5B is too small, the same code works with Qwen2.5-3B (just increase GPU memory)

### Why Not LoRA?

Full-parameter training (not LoRA/QLoRA) because:
- RL updates are subtle — LoRA's low-rank constraint may be too restrictive for learning new reasoning patterns
- 1.5B model fits comfortably in memory even without LoRA
- Simpler setup, fewer hyperparameters to tune

### Why math_verify Instead of String Matching?

Math answers can be expressed in many equivalent ways:
- `0.5` = `1/2` = `\frac{1}{2}` = `0.50`
- `\sqrt{2}/2` = `\frac{\sqrt{2}}{2}` = `\frac{1}{\sqrt{2}}`

`math_verify` uses SymPy to check symbolic equivalence, catching these cases.

## References

### Papers
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — The R1 and R1-Zero paper
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) — Introduces GRPO
- [DAPO: An Open-Source LLM Reinforcement Learning System](https://arxiv.org/abs/2503.14476) — beta=0 and other GRPO improvements
- [Dr. GRPO: Removing Bias from Group Relative Policy Optimization](https://arxiv.org/abs/2503.20783) — dr_grpo loss type, scale_rewards=False

### Tools
- [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl) — HuggingFace's RL training library
- [math-verify](https://github.com/huggingface/Math-Verify) — Robust math answer verification
- [MLX](https://github.com/ml-explore/mlx) — Apple Silicon ML framework

### Tutorials
- [Post-training an LLM for Reasoning with GRPO in TRL](https://huggingface.co/learn/cookbook/en/fine_tuning_llm_grpo_trl) — HuggingFace cookbook
- [Mini-R1: Reproduce Deepseek R1 "aha moment"](https://www.philschmid.de/mini-deepseek-r1) — Phil Schmid's tutorial
- [Implementing GRPO in TRL](https://huggingface.co/learn/llm-course/en/chapter12/4) — HuggingFace LLM course

## License

This project is for educational and research purposes.
