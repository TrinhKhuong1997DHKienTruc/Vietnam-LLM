import argparse
import os
import sys
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig  # Optional, only if bitsandbytes installed
except Exception:  # pragma: no cover - optional import
    BitsAndBytesConfig = None  # type: ignore


def detect_dtype(is_cuda: bool) -> torch.dtype:
    if is_cuda and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if is_cuda else torch.float32


def load_model(model_id: str, load_in_4bit: bool, device: Optional[str]):
    is_cuda = torch.cuda.is_available() and device in (None, "auto", "cuda")
    torch_dtype = detect_dtype(is_cuda)
    kwargs = dict(trust_remote_code=True)

    if load_in_4bit and BitsAndBytesConfig is not None:
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
            )
            return AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                quantization_config=bnb_config,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                **kwargs,
            )
        except Exception as e:
            print(f"[warn] 4-bit load failed: {e}. Falling back to non-quantized load.", file=sys.stderr)

    if is_cuda:
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            **kwargs,
        )

    # CPU fallback (offload to disk to reduce RAM peaks)
    offload_dir = os.path.join(os.getcwd(), "offload")
    os.makedirs(offload_dir, exist_ok=True)
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cpu",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        offload_folder=offload_dir,
        **kwargs,
    )


def run_local(model_id: str, prompt: str, max_new_tokens: int, temperature: float, top_p: float, load_in_4bit: bool, device: Optional[str]):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(model_id, load_in_4bit, device)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(text)


def run_inference_api(model_id: str, prompt: str, max_new_tokens: int, temperature: float, top_p: float, hf_token: Optional[str]):
    try:
        from huggingface_hub import InferenceClient
    except Exception as e:
        print("[error] huggingface-hub is required for --inference-api. Install requirements.", file=sys.stderr)
        raise e

    token = hf_token or os.environ.get("HUGGING_FACE_API_TOKEN")
    if not token:
        print("[error] Missing HUGGING_FACE_API_TOKEN. Set env var or pass --hf-token.", file=sys.stderr)
        sys.exit(2)
    client = InferenceClient(model=model_id, token=token)
    stream = client.text_generation(prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, stream=True)
    for chunk in stream:
        print(chunk, end="", flush=True)
    print()


def main():
    parser = argparse.ArgumentParser(description="Run TheFinAI/Fin-o1-8B demo")
    parser.add_argument("--model-id", default="TheFinAI/Fin-o1-8B")
    parser.add_argument("--prompt", default="You are Fin-o1. Answer briefly.\nWhat is 3-5? Explain the steps.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--inference-api", action="store_true", help="Use Hugging Face Inference API instead of local inference")
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    if args.inference_api:
        run_inference_api(args.model_id, args.prompt, args.max_new_tokens, args.temperature, args.top_p, args.hf_token)
    else:
        run_local(args.model_id, args.prompt, args.max_new_tokens, args.temperature, args.top_p, args.load_in_4bit, args.device)


if __name__ == "__main__":
    main()


