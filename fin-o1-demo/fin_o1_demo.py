#!/usr/bin/env python
import argparse
import os
import sys
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def create_generator(model_id: str, device_preference: Optional[str] = None):
	os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
	trust_remote_code = True

	if device_preference is None:
		if torch.cuda.is_available():
			device_preference = "cuda"
		elif torch.backends.mps.is_available():
			device_preference = "mps"
		else:
			device_preference = "cpu"

	if device_preference == "cuda":
		dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
		device_map = "auto"
	elif device_preference == "mps":
		dtype = torch.float16
		device_map = {"": "mps"}
	else:
		dtype = torch.float32
		device_map = {"": "cpu"}

	tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=trust_remote_code)
	model = AutoModelForCausalLM.from_pretrained(
		model_id,
		trust_remote_code=trust_remote_code,
		torch_dtype=dtype,
		device_map=device_map,
	)

	gen = pipeline(
		"text-generation",
		model=model,
		tokenizer=tokenizer,
		return_full_text=False,
		batch_size=1,
	)
	return gen


def main():
	parser = argparse.ArgumentParser(description="Run a quick Fin-o1-8B generation demo")
	parser.add_argument("--model-id", default="TheFinAI/Fin-o1-8B", help="Hugging Face model id")
	parser.add_argument("--prompt", default="List 3 short key takeaways from AAPL's latest earnings.", help="Prompt to generate from")
	parser.add_argument("--max-new-tokens", type=int, default=128)
	parser.add_argument("--temperature", type=float, default=0.7)
	parser.add_argument("--top-p", type=float, default=0.9)
	parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
	args = parser.parse_args()

	device_pref = None if args.device == "auto" else args.device

	print(f"Loading model: {args.model_id}", file=sys.stderr)
	generator = create_generator(args.model_id, device_preference=device_pref)

	print("\nPrompt:\n" + args.prompt)
	outputs = generator(
		args.prompt,
		max_new_tokens=args.max_new_tokens,
		do_sample=True,
		temperature=args.temperature,
		top_p=args.top_p,
		repetition_penalty=1.05,
	)
	text = outputs[0]["generated_text"] if outputs and isinstance(outputs, list) else str(outputs)
	print("\nModel output:\n" + text)


if __name__ == "__main__":
	main()