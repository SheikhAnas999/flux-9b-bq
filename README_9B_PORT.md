# FluxRT -> FLUX.2-klein-9B port (Modal / H100)

FluxRT (real-time webcam/video stream editing with a Spatial KV Cache)
retargeted from `FLUX.2-klein-4B` to `black-forest-labs/FLUX.2-klein-9B`
(the step-distilled 9B checkpoint), deployed on Modal with an H100.

## Which 9B, exactly (important)

BFL ships three 9B-family checkpoints:

- `FLUX.2-klein-9B`      -> step-distilled to 4 steps. USED HERE. The only
                            one viable for real-time few-step streaming.
- `FLUX.2-klein-base-9B` -> undistilled base. Needs ~50 steps + CFG. NOT
                            usable for real-time; do not point configs at it.
- `FLUX.2-klein-9b-kv`   -> KV-cache variant for repeated reference editing.
                            Not needed: FluxRT's SpatialCache already covers
                            reference-token reuse (mask value 2 once, then 0),
                            and its per-frame spatial reuse goes further than
                            the KV variant does.

If you say "base" meaning "the normal, non-KV model", `FLUX.2-klein-9B` is
the one you want, and it is what this repo uses.

## What was changed vs. upstream FluxRT (complete list)

1. `src/fluxrt/stream_processor/pipeline.py`
   The one real bug: `SpatialCache` was created with 4B-shaped defaults
   (5 double / 20 single layers, 24 heads). It now reads
   `self.transformer.config` at cache-creation time, so cache tensors match
   whatever checkpoint is loaded (9B family: 8 double / 24 single, 32 heads,
   head_dim 128, in_channels 128).

2. `configs/*.json`
   `models_path` -> `FLUX.2-klein-9B` in all five configs.
   `enable_int8_quantization` forced off (published int8 is 4B-only).

3. `src/fluxrt/stream_processor/model_inference_subprocess.py`
   int8 loader raises a clear error under a 9B models_path instead of
   silently loading mis-shaped 4B weights.

4. `modal_app.py`
   - Downloads `black-forest-labs/FLUX.2-klein-9B` (GATED: accept the FLUX
     Non-Commercial License on Hugging Face, then run
     `modal secret create huggingface-secret HF_TOKEN=hf_xxx`).
   - Serving GPU: `L40S` -> `H100` (80 GB). bf16 stack peaks ~45 GB:
     ~18 GB transformer + ~16 GB Qwen3-8B text encoder + VAE/RIFE +
     ~1.2 GB SpatialCache per active timestep + activations. No quantization
     needed on H100.

## Verified, no change needed

- Text-encoder layer taps (9, 18, 27): valid for the 9B family's Qwen3-8B
  embedder (36 layers, hidden 4096 -> joint_attention_dim 12288).
- Latent format in_channels=128, same FLUX.2 autoencoder: TAEF2 and
  FlowUpscaler stay compatible, no retraining.
- RIFE, shared-memory transport, GUI, virtual webcam, LivePortrait:
  model-agnostic, untouched.
- Dead `Flux2KVCache` code in transformer_flux2.py: still unused, harmless.

## Deploy on Modal

    # one-time
    modal secret create huggingface-secret HF_TOKEN=hf_xxx
    modal run modal_app.py::download_weights

    # serve
    modal deploy modal_app.py        # or: modal serve modal_app.py

First container start spends several minutes in torch.compile on the 9B
graph; the inductor/triton caches persist to the Modal volume so later cold
starts skip it.

## Sanity check on first run

`cat /weights/FLUX.2-klein-9B/transformer/config.json` and confirm
num_layers=8, num_single_layers=24, num_attention_heads=32. The SpatialCache
fix reads these dynamically, so even a different value will not crash --
this check is just to confirm expectations.

## Performance expectations

9B is ~2.25x the per-step FLOPs of 4B. On H100 at 368x640 / 2 steps with the
spatial cache active, expect roughly a third to half of 4B's FPS before
tuning. Levers, in order of impact: keep interpolation_exp 2 (perceived FPS
x4), lower resolution, enable_tiny_vae with taef2, raise the mask threshold
in update_controller.py so fewer tokens recompute per frame.

## Known gaps

- No fp8/int8 path for 9B in this repo. For a 32 GB consumer card, export
  your own optimum-quanto int8 (the 4B recipe in quantized_flux2.py shows
  the exact pattern).
- License: 9B checkpoints are FLUX Non-Commercial. Research/personal use ok;
  commercial use needs a BFL license.
