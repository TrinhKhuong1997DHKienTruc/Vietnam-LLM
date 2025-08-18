import os
import sys
import json
import time
from typing import Optional

MODEL_ID = os.environ.get("MODEL_ID", "TheFinAI/Fin-o1-8B")
DEFAULT_PROMPT = os.environ.get("FIN_O1_PROMPT", "What is the result of 3-5?")


def run_local(prompt: str, device_preference: Optional[str] = None) -> str:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=True,
        trust_remote_code=True,
    )

    device = "cpu"
    torch_dtype = None

    # Honor explicit device
    if device_preference in {"cpu", "cuda"}:
        device = device_preference

    # If CUDA available and not forced to CPU, try to use it
    try:
        import torch

        if device_preference != "cpu" and torch.cuda.is_available():
            device = "cuda"
            torch_dtype = torch.float16
    except Exception:
        pass

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto" if device == "cuda" else None,
    )

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    if device == "cuda":
        try:
            import torch
            input_ids = input_ids.to("cuda")
        except Exception:
            pass

    output = model.generate(
        input_ids,
        max_new_tokens=int(os.environ.get("MAX_NEW_TOKENS", "128")),
        do_sample=False,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def run_inference_api(prompt: str) -> str:
    # Try huggingface_hub client first; fall back to raw HTTP if needed
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for Inference API.")

    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=token)
        resp = client.text_generation(
            prompt,
            model=MODEL_ID,
            max_new_tokens=int(os.environ.get("MAX_NEW_TOKENS", "128")),
            temperature=0.0,
        )
        if isinstance(resp, str):
            return resp
    except Exception:
        pass

    # Raw HTTP fallback
    import requests
    url = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": int(os.environ.get("MAX_NEW_TOKENS", "128")), "temperature": 0.0},
    }
    r = requests.post(url, headers=headers, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
        return data[0]["generated_text"]
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"]
    return json.dumps(data)


def main():
    prompt = DEFAULT_PROMPT
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])

    use_api = os.environ.get("USE_INFERENCE_API", "0") == "1"
    device = os.environ.get("DEVICE")  # "cpu" or "cuda"

    t0 = time.time()
    try:
        if use_api:
            print("[demo] Using Hugging Face Inference API...")
            out = run_inference_api(prompt)
        else:
            print("[demo] Running locally with transformers...")
            out = run_local(prompt, device_preference=device)
    except Exception as e:
        print(f"[demo] Error: {e}")
        sys.exit(1)
    t1 = time.time()

    print("\n=== Prompt ===\n" + prompt)
    print("\n=== Output ===\n" + out)
    print(f"\n[demo] Elapsed: {t1 - t0:.2f}s")


if __name__ == "__main__":
    main()

