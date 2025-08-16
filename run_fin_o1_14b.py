#!/usr/bin/env python
"""Simple script to download TheFinAI/Fin-o1-14B from Hugging Face and run a sample generation.

Usage:
    python run_fin_o1_14b.py --prompt "Your prompt here" --max_new_tokens 128

Environment variables:
    HF_TOKEN   Optional. Your Hugging Face access token with permission to download the model.
"""
import argparse
import os
import sys
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from accelerate import init_empty_weights, infer_auto_device_map


DEFAULT_MODEL_ID = "TheFinAI/Fin-o1-14B"


def load_model(model_id: str = DEFAULT_MODEL_ID, hf_token: Optional[str] = None):
    """Load the model using 4/8-bit quantization if a GPU is available to save memory."""
    print(f"Loading '{model_id}' … This can take a while the first time.")

    # Detect device
    device_map = "auto"

    # Try to load in 8-bit to fit on a single modern GPU (needs bitsandbytes)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            load_in_8bit=True,
            device_map=device_map,
            torch_dtype=torch.float16,
            token=hf_token,
        )
    except Exception as err:
        print("\n⚠️  8-bit loading failed or not supported ({}). Falling back to full precision.".format(err))
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch.float16,
            token=hf_token,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, use_fast=True)
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Run inference with Fin-o1-14B model")
    parser.add_argument("--prompt", type=str, default="Explain the impact of interest rate changes on bond prices:", help="The prompt to feed into the model")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ID, help="Model id on Hugging Face Hub")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Number of tokens to generate")
    parser.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN"), help="Hugging Face token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    if args.hf_token is None:
        print("ℹ️ No HF token provided. If the model requires acceptance of license or gated access you may need to set HF_TOKEN.")

    model, tokenizer = load_model(args.model, args.hf_token)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print("\nGenerating…\n")
    outputs = pipe(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
    )

    print(outputs[0]["generated_text"])


if __name__ == "__main__":
    main()