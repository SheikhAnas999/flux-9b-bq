# Running FluxRT (9B port) on a RunPod pod instead of Modal

This mirrors what `modal_app.py` did, minus Modal's Image/Volume/Cls
machinery: same weights, same config, same `/stream` websocket protocol.
Only the hosting layer changes.

## 0. Rotate your HF token

You pasted your Hugging Face token in a chat earlier. Treat it as leaked —
revoke it at https://huggingface.co/settings/tokens and create a fresh one
before using any of this. Also make sure that account has accepted the FLUX
Non-Commercial License on the gated model page:
https://huggingface.co/black-forest-labs/FLUX.2-klein-9B

## 1. Create the pod

In the RunPod console:

1. **Deploy > GPU Pod**, pick an **H100 80GB** instance.
2. **Template**: a PyTorch template (e.g. "RunPod Pytorch 2.x") is a fine
   base — we install our own CUDA-matched torch build on top anyway, so the
   exact template CUDA version doesn't matter much.
3. **Storage**: attach your **Network Volume** and mount it at `/workspace`
   (this is where weights, the venv, and the repo will live so they survive
   pod stop/restart).
4. **Expose HTTP Ports**: add `8000` (or whatever port you'll run the server
   on) so RunPod's proxy can reach it later. You don't need the server
   running yet to set this.
5. Deploy the pod, then open a terminal via **Connect > Start Web Terminal**
   or SSH in directly.

## 2. Get the code onto the pod

From the pod terminal:

```bash
cd /workspace
git clone https://github.com/SheikhAnas999/flux-9b-bq
```

(Or `scp`/upload this modified copy — the important part is that
`runpod/`, `configs/runpod_config.json`, and the updated `client/index.html`
from this session end up under `/workspace/flux-9b-bq`.)

## 3. One-time environment setup

```bash
cd /workspace/flux-9b-bq
bash runpod/01_setup_env.sh
```

This installs system packages, creates a venv at `/workspace/venv` (on the
network volume, so it's not rebuilt every pod restart), and installs
torch/diffusers/transformers/etc. — the same package set `modal_app.py`
used.

## 4. Download the weights (one-time)

```bash
source /workspace/venv/bin/activate
export HF_TOKEN=hf_your_new_token
python runpod/02_download_weights.py
```

Pulls `FLUX.2-klein-9B`, `RIFE-safetensors`, `taef2`, and `FlowUpscaler`
into `/workspace/weights` on the network volume — roughly 20GB+, so this
only needs to happen once, ever (not on every pod restart).

## 5. Start the server (every session, manual)

```bash
source /workspace/venv/bin/activate
cd /workspace/flux-9b-bq
python runpod/server.py
```

First start spends several minutes in `torch.compile`-adjacent warmup on
the 9B graph, matching the same cold-start cost Modal had. Leave this
running in the terminal (or in `tmux`/`screen` so it survives you
disconnecting — see note below).

Health check from another terminal:

```bash
curl http://localhost:8000/health
```

## 6. Connect from the client

Find your pod's ID in the RunPod console ("Connect" panel or the URL bar),
then open `client/index.html` and confirm/edit:

```js
const WS_URL = "wss://<POD_ID>-8000.proxy.runpod.net/stream";
```

Open `client/index.html` in a browser (locally, or served any static way)
and it'll connect straight to your pod.

## Keeping it running across SSH disconnects

Since you're starting manually rather than auto-starting on boot, run it
inside `tmux` so closing your terminal doesn't kill the process:

```bash
tmux new -s fluxrt
source /workspace/venv/bin/activate && cd /workspace/flux-9b-bq && python runpod/server.py
# Ctrl+B, D to detach; `tmux attach -t fluxrt` to come back
```

## What's different from the Modal version, and why

| Modal concept | RunPod equivalent |
|---|---|
| `modal.Volume` | Network Volume mounted at `/workspace` |
| `modal.Image` (build-time deps) | `runpod/01_setup_env.sh` (run once, persisted in a venv on the volume) |
| `download_weights()` Modal function | `runpod/02_download_weights.py`, run manually |
| `FluxRTServer` class + `@modal.enter()`/`@modal.exit()` | plain `FastAPI` `on_startup`/`on_shutdown` hooks in `server.py` |
| `@modal.asgi_app()` | the same `FastAPI()` app, run directly with `uvicorn`/`python server.py` |
| Autoscaling (`min_containers=0`, `max_containers=1`) | none — a pod is a fixed machine you start/stop yourself |
| `modal secret create huggingface-secret` | `export HF_TOKEN=...` in your shell (or a RunPod pod env var) |

The `/stream` websocket protocol (JPEG frames in/out, `{"type": "prompt", ...}`
and `{"type": "reference", ...}` text commands) is untouched — the client
and `src/fluxrt/` package didn't need any changes, only the hosting glue did.
