# Fin-o1-8B Cross-Platform Demo

This folder provides ready-to-run scripts to download, setup, and demo `TheFinAI/Fin-o1-8B` on Windows and Linux/macOS.

## Quickstart (Linux/macOS)

```bash
cd fin-o1-demo
bash setup.sh            # create venv and install deps (CPU torch by default)
# Optional: pre-download model (can be large)
# bash setup.sh --download

# CLI demo
bash run_cli.sh "List 3 short key takeaways from AAPL's latest earnings."

# Gradio UI (http://127.0.0.1:7860)
bash run_gradio.sh

# Streamlit UI (http://127.0.0.1:8501)
bash run_streamlit.sh
```

## Quickstart (Windows)

```bat
cd fin-o1-demo
setup.bat               
REM Optional: pre-download model
REM setup.bat --download

REM CLI demo
run_cli.bat "List 3 short key takeaways from AAPL's latest earnings."

REM Gradio UI (http://127.0.0.1:7860)
run_gradio.bat

REM Streamlit UI (http://127.0.0.1:8501)
run_streamlit.bat
```

## Notes
- Default install uses CPU PyTorch for maximum compatibility. If you have CUDA, install a matching Torch build from `pytorch.org` and the scripts will auto-detect GPU.
- Set `HF_TOKEN` if the model requires authentication.
- Large model downloads: enable fast download via `HF_HUB_ENABLE_HF_TRANSFER=1` (already enabled in scripts).