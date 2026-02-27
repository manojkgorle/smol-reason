"""Convert HuggingFace model to MLX format for MacBook inference.

=== Why Convert to MLX? ===

MLX is Apple's machine learning framework optimized for Apple Silicon (M1-M4).
Converting from HuggingFace format to MLX format enables:

- **Unified memory**: Apple Silicon shares memory between CPU and GPU ("Neural Engine"),
  so a 1.5B model uses ~3GB in fp16, ~1.5GB in 4-bit quantization
- **Fast inference**: MLX generates ~100-200 tok/s on M4 Pro for 1.5B models
- **No GPU server needed**: run inference locally on your laptop

=== Quantization ===

Quantization reduces model size and speeds up inference by using fewer bits per weight:

- **Full precision (fp16)**: ~3GB for 1.5B params. Best quality, most memory.
- **8-bit**: ~1.5GB. Negligible quality loss. Good default.
- **4-bit**: ~0.8GB. Small quality loss. Best for memory-constrained devices.

For a 1.5B model, even fp16 fits easily on a MacBook with 16GB+ RAM,
so quantization is optional. For larger models (7B+), 4-bit is essential.

Usage:
    python inference/convert_model.py --model_path outputs/grpo-qwen2.5-1.5b
    python inference/convert_model.py --model_path outputs/grpo-qwen2.5-1.5b --quantize 4bit
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Convert HF model to MLX format")
    parser.add_argument("--model_path", type=str, required=True, help="HF model path or local dir")
    parser.add_argument(
        "--quantize",
        type=str,
        default=None,
        choices=["4bit", "8bit"],
        help="Quantization level",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: {model_path}-mlx)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or f"{args.model_path}-mlx"
    if args.quantize:
        output_dir = f"{output_dir}-{args.quantize}"

    # Build the mlx_lm.convert command
    # This tool handles: loading HF weights → converting to MLX tensors → saving
    cmd = [
        sys.executable, "-m", "mlx_lm.convert",
        "--hf-path", args.model_path,
        "--mlx-path", output_dir,
    ]

    if args.quantize == "4bit":
        cmd.extend(["--quantize", "--q-bits", "4"])
    elif args.quantize == "8bit":
        cmd.extend(["--quantize", "--q-bits", "8"])

    print(f"Converting {args.model_path} → {output_dir}")
    if args.quantize:
        print(f"Quantization: {args.quantize}")

    subprocess.run(cmd, check=True)
    print(f"Done! Model saved to {output_dir}")


if __name__ == "__main__":
    main()
