#!/usr/bin/env bash
# One-time setup on Linux / macOS:  bash scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
$PY -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA GPU detected - installing the CUDA build of PyTorch (cu130 covers RTX 50-series)"
  pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
else
  echo "No NVIDIA GPU detected — installing the smaller CPU build of PyTorch"
  pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
fi
pip install -r requirements.txt
[ -f .env ] || cp .env.example .env
(cd frontend && npm install && npm run build)
python -m training.download_data eurosat || echo "EuroSAT download failed — the studio can retry, or use synthetic data"
cat <<'MSG'

Setup complete.
  1. Edit .env: set ADMIN_PASSWORD (and GROQ_API_KEY for the AI agent, COPERNICUS_* or EE_PROJECT for real
     imagery, GOOGLE_MAPS_API_KEY for the Google 3D globe)
  2. Train:   source .venv/bin/activate && python -m training.pipeline          (quick demo)
              python -m training.pipeline --real                              (EuroSAT; GPU recommended)
  3. Run:     uvicorn main:app --port 8000   →  http://localhost:8000
MSG
