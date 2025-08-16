# Fin-o1-14B Quickstart

This repository shows a **minimal** example of how to download and run inference with the [TheFinAI/Fin-o1-14B](https://huggingface.co/TheFinAI/Fin-o1-14B) large language model using the Hugging Face `transformers` ecosystem.

> **Note**: The 14 B parameter model is **resource-intensive**. You will need a modern GPU with at least **24 GB of VRAM** (e.g. RTX 4090 / A6000) or to use 8-bit quantization. The included script will attempt 8-bit loading automatically. CPU inference is possible but extremely slow.

---

## 1. Install Dependencies

Create a fresh Python ≥ 3.9 environment (e.g. `conda`, `venv`, `pipenv`). Then install the required packages:

```bash
pip install -r requirements.txt
```

If you plan to run on GPU you must have **PyTorch** built with CUDA support. Visit [pytorch.org](https://pytorch.org) for the exact command matching your CUDA version.


## 2. Accept the Model License (once)

The Fin-o1-14B model is based on Meta Llama 2 and may be **gated** on the Hugging Face Hub. Make sure you are logged-in and have agreed to the license:

1. Go to the model page: <https://huggingface.co/TheFinAI/Fin-o1-14B>
2. Click *“Access repository”* and accept the terms.


## 3. Authenticate (optional but recommended)

If the model is gated you will need an **access token**. Create one from <https://huggingface.co/settings/tokens> and export it:

```bash
export HF_TOKEN="<your_token>"
```

The script will read the token from the `HF_TOKEN` environment variable or from the `--hf_token` flag.


## 4. Run Inference

### Default prompt

```bash
python run_fin_o1_14b.py
```

### Custom prompt and generation length

```bash
python run_fin_o1_14b.py \
  --prompt "How will a 25bps rate hike affect bank profitability?" \
  --max_new_tokens 200
```

### Specify a different model

The script is generic. Point to any compatible causal-LM on the Hub:

```bash
python run_fin_o1_14b.py --model meta-llama/Llama-2-7b-hf
```


## 5. Tips & Troubleshooting

* **Not enough memory?** The script first tries 8-bit (bitsandbytes) loading. You can also experiment with `load_in_4bit=True` (requires transformers ≥ 4.40).
* **Slow generation?** Use a GPU and set `torch_dtype=torch.float16` (already the default when a GPU is detected).
* **CUDA error about mismatched versions?** Make sure the PyTorch version you installed matches your local CUDA drivers.
* **License errors 403?** Ensure you accepted the model license with the same Hugging Face account used to create the token.


---

Happy experimenting! 🎉
