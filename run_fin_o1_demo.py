import os
import sys
import argparse
from typing import Optional

from rich import print
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


DEFAULT_MODEL = "TheFinAI/Fin-o1-8B"


def select_device(prefer_gpu: bool) -> torch.device:
	if prefer_gpu and torch.cuda.is_available():
		return torch.device("cuda")
	if prefer_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")


def load_model(model_name: str, device: torch.device, dtype: Optional[torch.dtype]) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
	print(f"[bold green]Loading model[/bold green]: {model_name} on device {device} ...")
	tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

	model_kwargs = {}
	if dtype is not None:
		model_kwargs["torch_dtype"] = dtype

	model = AutoModelForCausalLM.from_pretrained(
		model_name,
		low_cpu_mem_usage=True,
		device_map="auto" if device.type != "cpu" else None,
		**model_kwargs,
	)

	if device.type == "cpu":
		model = model.to(device)

	return tokenizer, model


def generate(tokenizer: AutoTokenizer, model: AutoModelForCausalLM, prompt: str, max_new_tokens: int = 256, temperature: float = 0.2) -> str:
	inputs = tokenizer(prompt, return_tensors="pt")
	inputs = {k: v.to(model.device) for k, v in inputs.items()}
	with torch.no_grad():
		output = model.generate(
			**inputs,
			max_new_tokens=max_new_tokens,
			temperature=temperature,
			early_stopping=True,
		)
	text = tokenizer.decode(output[0], skip_special_tokens=True)
	return text


def main():
	parser = argparse.ArgumentParser(description="Run Fin-o1-8B demo")
	parser.add_argument("--model", default=DEFAULT_MODEL, help="Model repo or local path")
	parser.add_argument("--prompt", default="You are Fin-o1. Explain briefly: What is the result of 3-5?", help="Prompt text")
	parser.add_argument("--gpu", action="store_true", help="Prefer GPU if available")
	parser.add_argument("--bf16", action="store_true", help="Use bfloat16 where supported")
	parser.add_argument("--fp16", action="store_true", help="Use float16 where supported")
	parser.add_argument("--max-new-tokens", type=int, default=128)
	args = parser.parse_args()

	device = select_device(args.gpu)
	dtype = None
	if args.bf16:
		dtype = torch.bfloat16
	elif args.fp16:
		dtype = torch.float16

	tokenizer, model = load_model(args.model, device, dtype)

	print("[bold cyan]Prompt:[/bold cyan]", args.prompt)
	response = generate(tokenizer, model, args.prompt, max_new_tokens=args.max_new_tokens)
	print("\n[bold yellow]Model Output:[/bold yellow]\n")
	print(response)


if __name__ == "__main__":
	main()