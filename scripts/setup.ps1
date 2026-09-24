# One-time setup on Windows (PowerShell):  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
  Write-Host "No NVIDIA GPU detected - installing the smaller CPU build of PyTorch"
  pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
}
pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
Push-Location frontend; npm install; npm run build; Pop-Location
try { python -m training.download_data eurosat } catch { Write-Host "EuroSAT download failed - retry from the studio or use synthetic data" }
Write-Host ""
Write-Host "Setup complete."
Write-Host "  1. Edit .env: set ADMIN_PASSWORD (and GROQ_API_KEY for the AI agent)"
Write-Host "  2. Train:  .\.venv\Scripts\Activate.ps1; python -m training.pipeline        (quick demo)"
Write-Host "             python -m training.pipeline --real                               (EuroSAT)"
Write-Host "  3. Run:    uvicorn main:app --port 8000   ->  http://localhost:8000"
