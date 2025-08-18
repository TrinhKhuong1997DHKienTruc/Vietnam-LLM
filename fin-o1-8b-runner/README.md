Fin-o1-8B Runner (Windows/Linux, Python 3.14 ready)

This project lets you run `TheFinAI/Fin-o1-8B` locally with `transformers` and `torch`, with fallbacks and scripts for quick setup on Windows or Linux. It also supports the Hugging Face Inference API if you prefer remote execution.

Quick start

- Windows (PowerShell):
  - `scripts\setup.bat`
  - `scripts\run_demo.bat`

- Linux/macOS:
  - `bash scripts/setup.sh`
  - `bash scripts/run_demo.sh`

Optional: pre-download weights locally

- Windows: `scripts\\download_model.bat`
- Linux/macOS: `bash scripts/download_model.sh`

Notes

- Python: Designed for Python 3.14. Works with 3.10–3.13 as well. If `torch` isn’t available for your Python build, the setup scripts automatically try a nightly wheel or CPU fallback.
- GPU: If you have NVIDIA CUDA, the setup tries to install a matching GPU `torch` by default. Otherwise it installs the CPU wheel.
- Memory: `Fin-o1-8B` is large. For local inference, you’ll typically want a modern GPU with ≥16 GB VRAM (more is better). CPU-only will be very slow and may require large RAM. If your hardware is insufficient, use the Inference API path below.

Features

- One-command setup on Windows and Linux
- Automatic `torch` wheel selection for Python 3.14 (stable → nightly → CPU fallbacks)
- Local `transformers` inference or remote Hugging Face Inference API
- Clean separation of dependencies: `torch` is installed by scripts to accommodate Python 3.14 wheels

Requirements

- Windows 10/11 or Linux
- Python 3.14 (recommended) or 3.10–3.13
- Pip and virtualenv available
- Optional: NVIDIA GPU with recent CUDA drivers

Setup

1) Download ZIP

- Click “Code” → “Download ZIP” on your GitHub branch, then unzip.

2) Install

- Windows: run `scripts\setup.bat` (double-click or from PowerShell)
- Linux/macOS: run `bash scripts/setup.sh`

The script:

- Creates a `.venv`
- Installs base deps from `requirements.txt`
- Installs `torch` with best-effort strategy for Python 3.14 (GPU wheel if available, otherwise CPU; falls back to nightly when needed)

3) Run a demo

- Windows: `scripts\run_demo.bat`
- Linux/macOS: `bash scripts/run_demo.sh`

Environment options (optional)

- `HF_TOKEN`: set if the model is gated or you want higher rate limits.
- `USE_INFERENCE_API=1`: use Hugging Face Inference API instead of local load.
- `FIN_O1_PROMPT`: override the default demo prompt.
- `MODEL_ID`: override Hugging Face repo ID (default `TheFinAI/Fin-o1-8B`).

Example (Linux):

```bash
export HF_TOKEN=hf_xxx   # if needed
export USE_INFERENCE_API=1
bash scripts/run_demo.sh
```

Troubleshooting

- Torch install fails on Python 3.14: setup falls back to nightly and/or CPU wheel. If all fail, re-run later or install a supported Python (3.12/3.13), then re-run `scripts/setup.*`.
- Out-of-memory (OOM): try `--device cpu` with small `max_new_tokens` (see `demo.py`), or switch to Inference API.
- Slow generation on CPU: expected for 8B models; prefer a GPU or the Inference API.

Security and privacy

- The project does not log prompts or outputs. If you use the Inference API, requests go to Hugging Face. Provide `HF_TOKEN` only if comfortable.

License

MIT

