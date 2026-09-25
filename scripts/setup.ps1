# One-time setup on Windows (PowerShell):  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# PyPI's Windows torch wheel is CPU-only, so pick the build explicitly. cu130 covers every current NVIDIA
# GPU including RTX 50-series (Blackwell needs CUDA 12.8 or newer).
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
  Write-Host "NVIDIA GPU detected - installing the CUDA build of PyTorch"
  pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
} else {
  Write-Host "No NVIDIA GPU detected - installing the smaller CPU build of PyTorch"
  pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
}
python -c "import torch; print('PyTorch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
Push-Location frontend; npm install; npm run build; Pop-Location
try { python -m training.download_data eurosat } catch { Write-Host "EuroSAT download failed - retry from the studio or use synthetic data" }
Write-Host ""
Write-Host "Setup complete."
Write-Host "  1. Edit .env: ADMIN_PASSWORD, COPERNICUS_CLIENT_ID/SECRET (Sentinel-2), CESIUM_ION_TOKEN (3D globe), GROQ_API_KEY (AI agent)"
Write-Host "  2. Train:  .\.venv\Scripts\Activate.ps1; python -m training.pipeline        (quick demo)"
Write-Host "             python -m training.pipeline --real                               (EuroSAT)"
Write-Host "  3. Run:    uvicorn main:app --port 8000   ->  http://localhost:8000"
