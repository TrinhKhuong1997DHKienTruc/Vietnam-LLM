# Fin-o1-14B Quickstart

This repository shows how to download and run [TheFinAI/Fin-o1-14B](https://huggingface.co/TheFinAI/Fin-o1-14B) locally with minimal hassle.

## Prerequisites

1. **Python 3.9 or newer**
2. **CUDA-enabled GPU** with **≥24 GB VRAM** is highly recommended. The script can load the model in 4-bit quantized mode (≈10-12 GB VRAM) via `bitsandbytes`, but CPU inference will be extremely slow.
3. **Git + Git-LFS** for downloading large weights from Hugging Face.

## Setup

Install the Python dependencies into a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Tip**: If you have `nvidia-driver` ≥ 525 you can use the pre-built wheels for PyTorch 2.1.
>
> On Apple Silicon you can omit `bitsandbytes` and run the model in bfloat16 (requires ≥36 GB unified memory).

Authenticate with Hugging Face (only needed once per machine):

```bash
huggingface-cli login
```

## Running Inference

```bash
python run_fin_o1_14b.py --prompt "List three risks of quantitative easing."
```

Flags:

* `--no_4bit` – load the full-precision weights (requires ≥55 GB VRAM)
* `--max_new_tokens` – limit the length of the generated answer

The first run will download ~25 GB of model shards to `~/.cache/huggingface`. Subsequent runs start instantly.

## Troubleshooting

* **Out-of-memory error** → ensure 4-bit mode is enabled or use a GPU with more VRAM.
* **Slow throughput** → lower `max_new_tokens` or disable sampling (`--temperature 0`).

Feel free to tailor the generation parameters inside `run_fin_o1_14b.py` for your use-case.
