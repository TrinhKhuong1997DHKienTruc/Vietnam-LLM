"""
run_fin_o1_14b.py
=================
Simple command-line utility to run inference with the `TheFinAI/Fin-o1-14B` model from 🤗 Hugging Face.

The script automatically detects GPUs and supports 4-bit quantization via bitsandbytes for dramatically
reduced memory footprint. It falls back to bfloat16 full-precision weights if `load_in_4bit` is disabled.

Example usage:
    python run_fin_o1_14b.py --prompt "Explain the CAPM model in simple terms."
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

MODEL_NAME = "TheFinAI/Fin-o1-14B"

# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------

def _can_use_4bit() -> bool:
    """Return True if we can reasonably attempt 4-bit loading on the current hw."""
    return torch.cuda.is_available()


def get_bnb_config() -> BitsAndBytesConfig | None:
    """Return a BitsAndBytesConfig if 4-bit is desired, else None."""
    if not _can_use_4bit():
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


# -----------------------------------------------------------------------------
# Loading utilities
# -----------------------------------------------------------------------------

def load_model_and_tokenizer(load_in_4bit: bool = True):
    """Load Fin-o1-14B and its tokenizer, optionally in 4-bit quantized mode."""

    quant_cfg = get_bnb_config() if load_in_4bit else None

    print("⏳ Loading model… (this may take a while on first run)")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=quant_cfg,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print("✅ Model loaded!")
    return model, tokenizer


# -----------------------------------------------------------------------------
# Generation helper
# -----------------------------------------------------------------------------

def generate(
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    load_in_4bit: bool = True,
) -> str:
    """Generate a response for *prompt* using Fin-o1-14B."""

    model, tokenizer = load_model_and_tokenizer(load_in_4bit)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# -----------------------------------------------------------------------------
# CLI entry-point
# -----------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with Fin-o1-14B")
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        required=True,
        help="Text prompt to send to the model.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--no_4bit",
        action="store_false",
        dest="load_in_4bit",
        help="Disable 4-bit quantized loading (use full precision).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)

    response = generate(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        load_in_4bit=args.load_in_4bit,
    )

    print("\n=== Response ===\n")
    print(response)


if __name__ == "__main__":
    main()