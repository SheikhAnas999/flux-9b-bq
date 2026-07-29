"""
One-time (idempotent) download of FluxRT's model weights onto the RunPod
Network Volume. This is the RunPod equivalent of modal_app.py's
`download_weights` function -- same repos, same target layout, just a plain
script instead of a Modal function decorator.

Usage:
    export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    python runpod/02_download_weights.py

Notes:
  - black-forest-labs/FLUX.2-klein-9B is GATED. Before this will work, log
    into huggingface.co with the account that owns HF_TOKEN and accept the
    FLUX Non-Commercial License on the model page:
    https://huggingface.co/black-forest-labs/FLUX.2-klein-9B
  - Because weights land on the Network Volume at /workspace/weights, you
    only need to run this once -- it persists across pod restarts/stops.
  - Re-running is safe: snapshot_download skips files that are already
    downloaded and complete.
"""

import os

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
# xet backend (hf_xet) has a known "Unable to parse string as hex hash
# value" bug on some repos; disable it and fall back to hf_transfer/http.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import snapshot_download  # noqa: E402

WEIGHTS_DIR = "/workspace/weights"

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise SystemExit(
        "HF_TOKEN env var is not set.\n"
        "Run: export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    )

os.makedirs(WEIGHTS_DIR, exist_ok=True)

print("Downloading black-forest-labs/FLUX.2-klein-9B (gated, needs HF_TOKEN + accepted license)...")
snapshot_download(
    "black-forest-labs/FLUX.2-klein-9B",
    local_dir=f"{WEIGHTS_DIR}/FLUX.2-klein-9B",
    token=hf_token,
)

print("Downloading TensorForger/RIFE-safetensors ...")
snapshot_download(
    "TensorForger/RIFE-safetensors",
    local_dir=f"{WEIGHTS_DIR}/RIFE-safetensors",
)

print("Downloading madebyollin/taef2 ...")
snapshot_download(
    "madebyollin/taef2",
    local_dir=f"{WEIGHTS_DIR}/taef2",
)

print("Downloading TensorForger/FlowUpscaler ...")
snapshot_download(
    "TensorForger/FlowUpscaler",
    local_dir=f"{WEIGHTS_DIR}/FlowUpscaler",
)

print(f"\nDone. Weights are on the network volume at {WEIGHTS_DIR}")
print("Sanity check the 9B config with:")
print(f"  cat {WEIGHTS_DIR}/FLUX.2-klein-9B/transformer/config.json")
print("  (expect num_layers=8, num_single_layers=24, num_attention_heads=32)")
