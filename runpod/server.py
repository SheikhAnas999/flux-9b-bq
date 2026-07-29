"""
FastAPI server for FluxRT's StreamProcessor, meant to run on a RunPod GPU
pod (started manually over SSH). This is a direct port of modal_app.py's
FluxRTServer -- same StreamProcessor, same /stream websocket protocol --
just without Modal's Image/Volume/Cls wrapping, since on a pod the process
just runs directly on the machine.

Run it:
    source /workspace/venv/bin/activate
    export HF_TOKEN=hf_xxxx      # only needed if weights aren't downloaded yet
    python runpod/server.py

Then, from outside, connect to:
    wss://<POD_ID>-8000.proxy.runpod.net/stream
(<POD_ID> is shown in the RunPod console; 8000 must be added as an exposed
HTTP port on the pod. See runpod/README.md.)
"""

import base64
import json
import os
import sys
import time

WEIGHTS_DIR = "/workspace/weights"
APP_DIR = "/workspace/flux-9b-bq"
CONFIG_PATH = f"{APP_DIR}/configs/runpod_config.json"

# fluxrt isn't pip-installed (no pyproject.toml ships in this repo) -- add
# src/ to the path directly, same trick modal_app.py does via PYTHONPATH.
sys.path.insert(0, f"{APP_DIR}/src")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402

sp = None  # global StreamProcessor, set during startup


def _startup():
    global sp

    if not os.path.isdir(WEIGHTS_DIR):
        raise SystemExit(f"{WEIGHTS_DIR} not found. Run runpod/02_download_weights.py first.")

    # fluxrt resolves all model/weight paths relative to the current working
    # directory (models_path, "RIFE-safetensors/...", "taef2/...",
    # "FlowUpscaler/..."), so chdir into the weights dir first -- exactly
    # what modal_app.py does with os.chdir(WEIGHTS_DIR).
    os.chdir(WEIGHTS_DIR)

    from fluxrt import StreamProcessor

    print("Loading StreamProcessor...")
    sp = StreamProcessor(CONFIG_PATH)
    sp.start()

    print("Warming up (loading models + first compiled forward pass)... this can take several minutes.")
    while not sp.is_ready():
        time.sleep(0.5)
    print("Ready.")


def _shutdown():
    if sp is not None:
        sp.stop()


app = FastAPI(on_startup=[_startup], on_shutdown=[_shutdown])


@app.get("/health")
async def health():
    return {"status": "ready" if (sp is not None and sp.is_ready()) else "loading"}


@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()

    from fluxrt.utils import crop_maximal_rectangle

    resolution = sp.get_resolution()
    height, width = resolution["height"], resolution["width"]
    input_tensor = sp.get_input_tensor()
    output_tensor = sp.get_output_tensor()

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
                ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
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
                    sp.set_prompt(cmd["value"])
                elif cmd.get("type") == "reference":
                    data = cmd.get("data")
                    if data is None:
                        sp.set_reference_image(None)
                    else:
                        raw = base64.b64decode(data)
                        arr = np.frombuffer(raw, dtype=np.uint8)
                        ref = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if ref is not None:
                            ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
                            sp.set_reference_image(ref)
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
