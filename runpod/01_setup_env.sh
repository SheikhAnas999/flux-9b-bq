#!/bin/bash
# One-time environment setup for FluxRT on a RunPod H100 pod.
#
# Prerequisites (do these in the RunPod web console before running this):
#   1. Create/attach a Network Volume, mounted at /workspace.
#   2. Put the flux-9b-bq repo at /workspace/flux-9b-bq (git clone or upload+unzip).
#
# Usage (from an SSH session into the pod):
#   cd /workspace/flux-9b-bq
#   bash runpod/01_setup_env.sh
#
# Safe to re-run — every step is idempotent.

set -euo pipefail

WEIGHTS_DIR="/workspace/weights"
APP_DIR="/workspace/flux-9b-bq"
VENV_DIR="/workspace/venv"

GREEN='\033[0;32m'; NC='\033[0m'
log() { echo -e "${GREEN}[+]${NC} $*"; }

[ -d "$APP_DIR" ] || { echo "ERROR: $APP_DIR not found. Clone/upload the repo there first."; exit 1; }

log "System packages..."
apt-get update -qq
apt-get install -y -qq git libgl1 libglib2.0-0 build-essential

log "Python venv at ${VENV_DIR} (persists on the network volume, so this only happens once)..."
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
fi
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip -q

log "PyTorch (CUDA 12.8, matches the tested Modal stack)..."
python -c "import torch" 2>/dev/null || \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 -q

log "Python deps (same set as modal_app.py's image)..."
pip install -q \
    triton \
    diffusers \
    transformers \
    "numpy>=2.0" \
    opencv-python-headless==4.12.0.88 \
    accelerate \
    "safetensors>=0.4.0" \
    optimum-quanto \
    peft \
    "fastapi[standard]" \
    "huggingface_hub[hf_transfer]"

# NOTE: there's no pyproject.toml in this repo (install.sh expects one, but
# it isn't shipped), so we don't `pip install -e .`. Instead server.py adds
# src/ to sys.path directly -- the exact same trick modal_app.py uses via
# PYTHONPATH.

mkdir -p "${WEIGHTS_DIR}"
mkdir -p "${WEIGHTS_DIR}/compile-cache/inductor" "${WEIGHTS_DIR}/compile-cache/triton"

log "Done."
echo
echo "Next steps, each time you start a fresh session:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  export HF_TOKEN=hf_xxxx                     # first time only, see step 2"
echo "  python runpod/02_download_weights.py         # first time only, ~20GB+ download"
echo "  python runpod/server.py                      # start the FastAPI server"
