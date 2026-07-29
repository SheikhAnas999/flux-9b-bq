"""
Deploys FluxRT's StreamProcessor behind a WebSocket endpoint on Modal.

Usage:
    modal run modal_app.py::download_weights   # one-time: pull weights into the Volume
    modal deploy modal_app.py                  # deploy the websocket server
    modal serve modal_app.py                   # dev mode with hot reload

The FluxRT package itself (src/fluxrt) is untouched -- this file only wraps
StreamProcessor with Modal's Image/Volume/Cls primitives and a FastAPI
websocket route.
"""

import os

import modal

WEIGHTS_DIR = "/weights"
APP_DIR = "/app"
CONFIG_PATH = f"{APP_DIR}/configs/modal_config.json"

app = modal.App("fluxrt-stream")

weights_volume = modal.Volume.from_name("fluxrt-weights", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git", "libgl1", "libglib2.0-0", "build-essential")
    .pip_install(
        "torch", "torchvision", index_url="https://download.pytorch.org/whl/cu128"
    )
    .pip_install(
        "triton",
        "diffusers",
        "transformers",
        "numpy>=2.0",
        "opencv-python-headless==4.12.0.88",
        "accelerate",
        "safetensors>=0.4.0",
        "optimum-quanto",
        "peft",
        "fastapi[standard]",
    )
    .env(
        {
            "TORCHINDUCTOR_CACHE_DIR": f"{WEIGHTS_DIR}/compile-cache/inductor",
            "TRITON_CACHE_DIR": f"{WEIGHTS_DIR}/compile-cache/triton",
        }
    )
    .env({"PYTHONPATH": f"{APP_DIR}/src"})
    .add_local_dir("src", remote_path=f"{APP_DIR}/src", copy=True)
    .add_local_dir("configs", remote_path=f"{APP_DIR}/configs", copy=True)
)

download_image = modal.Image.debian_slim().pip_install("huggingface_hub[hf_transfer]").env(
    # xet backend (hf_xet) has a known "Unable to parse string as hex hash
    # value" bug on some repos; disable it and fall back to hf_transfer/http.
    {"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_DISABLE_XET": "1"}
)


@app.function(
    image=download_image,
    volumes={WEIGHTS_DIR: weights_volume},
    timeout=3600,
    # klein-9B is a GATED repo (FLUX Non-Commercial License). Accept the
    # license on huggingface.co first, then create the secret once with:
    #   modal secret create huggingface-secret HF_TOKEN=hf_xxx
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download_weights():
    """One-time job: pull FLUX.2-klein-9B (base) and RIFE weights into the Modal Volume."""
    from huggingface_hub import snapshot_download

    hf_token = os.environ.get("HF_TOKEN")

    print("Downloading black-forest-labs/FLUX.2-klein-9B (gated, needs HF_TOKEN)...")
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

    weights_volume.commit()
    print("Done. Weights committed to volume.")


@app.cls(
    image=image,
    # klein-9B bf16 stack needs ~45GB peak (~18GB transformer + ~16GB Qwen3-8B
    # text encoder + VAE/RIFE + per-timestep SpatialCache + activations).
    # L40S (48GB) is too tight and too slow for real-time; H100 80GB is the
    # sweet spot. "H200" gives headroom for higher res / flow upscaler.
    gpu="H100",
    volumes={WEIGHTS_DIR: weights_volume},
    min_containers=0,
    max_containers=1,  # cap fan-out so a client reconnect loop can't spin up parallel H100s
    scaledown_window=30,
    startup_timeout=30 * 60,
)
@modal.concurrent(max_inputs=1)
class FluxRTServer:
    @modal.enter()
    def load(self):
        import sys
        import time

        os.chdir(WEIGHTS_DIR)
        sys.path.insert(0, f"{APP_DIR}/src")

        from fluxrt import StreamProcessor

        self.sp = StreamProcessor(CONFIG_PATH)
        self.sp.start()

        print("Warming up (loading models + first compiled forward pass)...")
        while not self.sp.is_ready():
            time.sleep(0.5)
        print("Ready.")
        # Persist inductor/triton compile caches so future cold starts skip
        # the multi-minute torch.compile.
        weights_volume.commit()

    @modal.exit()
    def unload(self):
        self.sp.stop()

    @modal.asgi_app()
    def web(self):
        import json

        import cv2
        import numpy as np
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect

        sys_path_marker = f"{APP_DIR}/src"
        import sys

        if sys_path_marker not in sys.path:
            sys.path.insert(0, sys_path_marker)
        from fluxrt.utils import crop_maximal_rectangle

        web_app = FastAPI()

        @web_app.websocket("/stream")
        async def stream(ws: WebSocket):
            await ws.accept()
            resolution = self.sp.get_resolution()
            height, width = resolution["height"], resolution["width"]
            input_tensor = self.sp.get_input_tensor()
            output_tensor = self.sp.get_output_tensor()

            try:
                while True:
                    msg = await ws.receive()

                    frame_bytes = msg.get("bytes")
                    if frame_bytes is not None:
                        arr = np.frombuffer(frame_bytes, dtype=np.uint8)
                        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if frame is None:
                            continue
                        frame = crop_maximal_rectangle(frame, height, width)
                        input_tensor.copy_from(frame)

                        out = output_tensor.to_numpy()
                        ok, buf = cv2.imencode(
                            ".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80]
                        )
                        if ok:
                            await ws.send_bytes(buf.tobytes())
                        continue

                    text = msg.get("text")
                    if text is not None:
                        try:
                            cmd = json.loads(text)
                        except json.JSONDecodeError:
                            continue
                        if cmd.get("type") == "prompt" and "value" in cmd:
                            self.sp.set_prompt(cmd["value"])
                        elif cmd.get("type") == "reference":
                            data = cmd.get("data")
                            if data is None:
                                self.sp.set_reference_image(None)
                            else:
                                import base64

                                raw = base64.b64decode(data)
                                arr = np.frombuffer(raw, dtype=np.uint8)
                                ref = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                                if ref is not None:
                                    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
                                    self.sp.set_reference_image(ref)
            except WebSocketDisconnect:
                pass

        return web_app
