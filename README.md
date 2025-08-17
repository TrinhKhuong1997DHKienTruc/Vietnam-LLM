# Vietnam-LLM: Fin-o1-8B Demo

This repository provides a ready-to-run demo for `TheFinAI/Fin-o1-8B` using Hugging Face Transformers. It works on Linux and Windows.

## Quickstart (Linux & Windows)

1. Install Python 3.13.x (3.13.7 requested). If you face package issues on 3.13, use Python 3.12.x.
2. Open a terminal in the repo folder (or unzip the GitHub ZIP and open that folder).
3. Create a virtual environment:
   - Linux/macOS:
     ```bash
     python3 -m venv .venv && source .venv/bin/activate
     ```
   - Windows (PowerShell):
     ```powershell
     py -m venv .venv; .\.venv\Scripts\Activate.ps1
     ```
4. Upgrade pip and install requirements:
   ```bash
   python -m pip install -U pip
   pip install -r requirements.txt
   ```
5. Run the demo:
   ```bash
   python run_fin_o1_demo.py --gpu  # add --gpu if you have CUDA/MPS
   ```

Example:
```bash
python run_fin_o1_demo.py --prompt "Briefly, what is 3-5?"
```

The script will download the model from Hugging Face and print the generated output.

## Notes on Dependencies

- PyTorch wheels differ by OS/CUDA. The `requirements.txt` allows pip to pick the right build.
- If `torch` installation fails on Python 3.13, try Python 3.12 or install a platform wheel from `pytorch.org`.
- Optional: pass `--bf16` or `--fp16` for reduced precision on supported GPUs.

## Model
- Model: `TheFinAI/Fin-o1-8B` on Hugging Face.
- Loaded via `transformers` `AutoModelForCausalLM` + `AutoTokenizer`.

## License
This repo is under GPL-2.0 (see `LICENSE`). The model weights and license are controlled by their respective owners on Hugging Face.
