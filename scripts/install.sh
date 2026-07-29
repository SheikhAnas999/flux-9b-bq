#!/bin/bash
# FluxRT installation script.
# Run from the repository root: bash scripts/install.sh
set -euo pipefail

# ── colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

log()  { echo -e "${GREEN}[+]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
die()  { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── sanity-check: running from repo root ──────────────────────────────────────
[ -f "pyproject.toml" ] || die "This script must be run from the FluxRT repository root."

# ── prerequisites ─────────────────────────────────────────────────────────────
log "Checking prerequisites..."

command -v git   &>/dev/null || die "'git' is not installed. Install it with your system package manager."
command -v conda &>/dev/null || die "'conda' is not installed. Install Miniconda or Anaconda first."
git lfs version  &>/dev/null || die "'git-lfs' is not installed. Install with: sudo apt install git-lfs  (or brew install git-lfs)"

log "All prerequisites found."

# ── conda environment ─────────────────────────────────────────────────────────
CONDA_ENV="fluxrt"

# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | grep -qE "^${CONDA_ENV}[[:space:]]"; then
    log "Conda environment '${CONDA_ENV}' already exists."
else
    log "Creating conda environment '${CONDA_ENV}' (python=3.12)..."
    conda create -n "$CONDA_ENV" python=3.12 pip -y
fi

conda activate "$CONDA_ENV"

# ── PyTorch ───────────────────────────────────────────────────────────────────
if python -c "import torch" 2>/dev/null; then
    log "PyTorch is already installed."
else
    log "Installing PyTorch with CUDA 12.8 support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
fi

# ── Python requirements ───────────────────────────────────────────────────────
# Use 'diffusers' as a proxy — it's the heaviest transitive dependency.
if python -c "import diffusers" 2>/dev/null; then
    log "Python requirements already installed."
else
    log "Installing Python requirements from requirements.txt..."
    pip install -r requirements.txt
fi

# ── fluxrt package ────────────────────────────────────────────────────────────
if python -c "import fluxrt" 2>/dev/null; then
    log "fluxrt package already installed."
else
    log "Installing fluxrt package in editable mode..."
    pip install -e .
fi

# ── model downloads ───────────────────────────────────────────────────────────
# Register LFS hooks for the current user (idempotent).
git lfs install

# clone_or_resume <dir> <url> <sentinel-file> <label>
#   sentinel-file — a large LFS asset that only exists after a complete download.
#   If the directory is present but the sentinel is missing we assume the clone
#   was interrupted and attempt to resume via `git lfs pull`.
clone_or_resume() {
    local dir="$1"
    local url="$2"
    local sentinel="$3"
    local label="$4"

    if [ -f "$sentinel" ]; then
        log "${label}: already downloaded."
        return
    fi

    if [ -d "${dir}/.git" ]; then
        warn "${label}: directory exists but looks incomplete — resuming LFS download..."
        git -C "$dir" pull --ff-only
        git -C "$dir" lfs pull
    elif [ -d "$dir" ]; then
        warn "${label}: directory '${dir}' exists but is not a git repository." \
             "Remove it and re-run to download the model."
        return
    else
        log "Downloading ${label}..."
        git clone "$url" "$dir"
    fi
}

clone_or_resume \
    "RIFE-safetensors" \
    "https://huggingface.co/TensorForger/RIFE-safetensors" \
    "RIFE-safetensors/flownet.safetensors" \
    "RIFE frame-interpolation model"

clone_or_resume \
    "FLUX.2-klein-4B" \
    "https://huggingface.co/black-forest-labs/FLUX.2-klein-4B" \
    "FLUX.2-klein-4B/transformer/diffusion_pytorch_model.safetensors" \
    "FLUX.2-klein-4B base model"

clone_or_resume \
    "FLUX.2-klein-4B-int8" \
    "https://huggingface.co/aydin99/FLUX.2-klein-4B-int8" \
    "FLUX.2-klein-4B-int8/diffusion_pytorch_model.safetensors" \
    "FLUX.2-klein-4B int8 model"

# ── done ──────────────────────────────────────────────────────────────────────
echo ""
log "${BOLD}Installation complete.${NC}"
log "Activate the environment and start:  ${BOLD}conda activate ${CONDA_ENV}${NC}"
log "Then run, for example:               ${BOLD}python scripts/run_gradio_demo.py${NC}"
