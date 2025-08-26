import argparse
import os
import sys
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "TheFinAI/Fin-o1-8B"


def _maybe_get_4bit_quant() -> Tuple[bool, dict]:
	"""Try enabling 4-bit quantization if bitsandbytes and a CUDA GPU are available."""
	if not torch.cuda.is_available():
		return False, {}
	try:
		# Only try to import when CUDA is present
		import bitsandbytes as bnb  # noqa: F401
		from transformers import BitsAndBytesConfig

		compute_dtype = (
			torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
		)
		quant_config = BitsAndBytesConfig(
			load_in_4bit=True,
			bnb_4bit_quant_type="nf4",
			bnb_4bit_compute_dtype=compute_dtype,
			bnb_4bit_use_double_quant=True,
		)
		return True, {"quantization_config": quant_config}
	except Exception:
		return False, {}


def load_model_and_tokenizer(prefer_gpu: bool = True):
	load_kwargs = {
		"device_map": "auto",
		"low_cpu_mem_usage": True,
		"trust_remote_code": True,
	}

	# Attempt 4-bit on GPU if present
	enable_4bit, extra = _maybe_get_4bit_quant() if prefer_gpu else (False, {})
	load_kwargs.update(extra)

	# Dtype selection: prefer bf16 on capable GPUs, else fp16 on GPU, else fp32 on CPU
	if not enable_4bit:
		if torch.cuda.is_available():
			torch_dtype = (
				torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
			)
		else:
			torch_dtype = torch.float32
		load_kwargs["torch_dtype"] = torch_dtype

	# Optional attention implementation override
	attn_impl = os.environ.get("ATTN_IMPL")
	if attn_impl:
		load_kwargs["attn_implementation"] = attn_impl

	print(f"Loading tokenizer: {MODEL_ID}")
	tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True)

	print(
		f"Loading model: {MODEL_ID} (4bit={'yes' if enable_4bit else 'no'}, dtype="
		f"{load_kwargs.get('torch_dtype', 'auto')})"
	)
	model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)

	return tokenizer, model


def build_prompt(tokenizer, user_prompt: str) -> str:
	# Prefer chat template if the tokenizer provides one
	try:
		messages = [{"role": "user", "content": user_prompt}]
		prompt_text = tokenizer.apply_chat_template(
			messages, tokenize=False, add_generation_prompt=True
		)
		return prompt_text
	except Exception:
		return f"User: {user_prompt}\nAssistant:"


def run_inference(user_prompt: str, max_new_tokens: int, seed: int) -> str:
	torch.manual_seed(seed)

	tokenizer, model = load_model_and_tokenizer()

	# Warn about CPU memory if running on CPU
	if not torch.cuda.is_available():
		print(
			"[Warning] CUDA GPU not detected. Running on CPU may require >= 32GB RAM and be slow.",
			file=sys.stderr,
		)

	prompt_text = build_prompt(tokenizer, user_prompt)
	inputs = tokenizer(prompt_text, return_tensors="pt")
	inputs = {k: v.to(model.device) for k, v in inputs.items()}

	with torch.no_grad():
		output_ids = model.generate(
			**inputs,
			max_new_tokens=max_new_tokens,
			do_sample=True,
			temperature=0.7,
			top_p=0.95,
			eos_token_id=tokenizer.eos_token_id,
		)

	output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
	return output_text


def main():
	parser = argparse.ArgumentParser(description="Run Fin-o1-8B demo inference")
	parser.add_argument(
		"--prompt",
		type=str,
		default="Hãy giải thích ngắn gọn: 3 - 5 bằng bao nhiêu?",
		help="User prompt (Vietnamese description is fine)",
	)
	parser.add_argument("--max-new-tokens", type=int, default=64)
	parser.add_argument("--seed", type=int, default=42)
	args = parser.parse_args()

	print("Starting inference... This may download model weights on first run.")
	output_text = run_inference(args.prompt, args.max_new_tokens, args.seed)

	print("\n=== Model output ===\n")
	print(output_text)


if __name__ == "__main__":
	main()
