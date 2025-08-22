Param(
  [switch]$GPU
)

Write-Host "Creating venv (.venv)"
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

if ($GPU) {
  Write-Host "Installing PyTorch GPU (use the official selector if this fails)"
  python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
} else {
  Write-Host "Installing PyTorch CPU"
  python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
}

Write-Host "Installing requirements"
python -m pip install -r requirements.txt

Write-Host "Done. Activate with: . .\\.venv\\Scripts\\Activate.ps1"

