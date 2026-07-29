"""
One-time (idempotent) fp8 quantization of the FLUX.2-klein-9B transformer +
text encoder, producing a checkpoint in the exact layout
`StreamProcessor.load_quantized_models()` expects. This is what makes 9B fit
on a 24GB card (e.g. RTX 3090) instead of needing an 80GB H100/A100.

Must run AFTER runpod/02_download_weights.py (needs the bf16 FLUX.2-klein-9B
weights already on the network volume).

Usage:
    source /workspace/venv/bin/activate
    python runpod/03_quantize_fp8.py

Notes:
  - Transformer and text encoder are quantized SEQUENTIALLY, each loaded
    alone in bf16, quantized, saved, then freed -- so peak VRAM is ~18GB
    (transformer) then ~16.4GB (text encoder), never both at once. This
    fits on a 24GB 3090 even though bf16-transformer + bf16-text-encoder
    together (~34GB) would not.
  - VAE is intentionally NOT quantized here. load_quantized_models() never
    reads a VAE from int8_models_path -- it always loads the VAE from the
    original bf16 models_path regardless of enable_int8_quantization. A
    quantized VAE would just be dead weight on disk.
  - Output layout (all paths relative to WEIGHTS_DIR):
      FLUX.2-klein-9B-fp8/
      ├── config.json                          (transformer config, flat)
      ├── diffusion_pytorch_model.safetensors   (quantized transformer, flat)
      ├── quanto_qmap.json                      (transformer quant map, flat)
      ├── text_encoder/
      │   ├── config.json
      │   ├── model.safetensors                (single file -- forced, not sharded)
      │   └── quanto_qmap.json
      └── tokenizer/                            (plain copy, nothing quantized)
  - Re-running is safe-ish: each stage is skipped if its output already
    exists, so an interrupted run can be resumed. Delete OUT_DIR to force
    a full redo.
"""

import json
import os
import sys

APP_DIR = "/workspace/flux-9b-bq"
WEIGHTS_DIR = "/workspace/weights"
MODELS_PATH = f"{WEIGHTS_DIR}/FLUX.2-klein-9B"
OUT_DIR = f"{WEIGHTS_DIR}/FLUX.2-klein-9B-fp8"

# fluxrt isn't pip-installed -- same sys.path trick server.py / modal_app.py use.
sys.path.insert(0, f"{APP_DIR}/src")

if not os.path.isdir(MODELS_PATH):
    raise SystemExit(
        f"{MODELS_PATH} not found. Run runpod/02_download_weights.py first."
    )

import torch  # noqa: E402
from optimum.quanto import qfloat8, quantize, freeze, quantization_map  # noqa: E402


def quantize_transformer():
    marker = f"{OUT_DIR}/diffusion_pytorch_model.safetensors"
    if os.path.exists(marker):
        print(f"[skip] transformer already quantized -> {marker}")
        return

    print("Loading bf16 transformer...")
    from fluxrt.stream_processor.transformer_flux2 import Flux2Transformer2DModel
    from fluxrt.stream_processor.quantized_flux2 import (
        QuantizedFlux2Transformer2DModel,
    )

    transformer = Flux2Transformer2DModel.from_pretrained(
        f"{MODELS_PATH}/transformer", local_files_only=True
    ).to("cuda", torch.bfloat16)

    print(f"Peak VRAM before quantize: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    print("Quantizing transformer to fp8 (weights-only, no calibration needed)...")
    qmodel = QuantizedFlux2Transformer2DModel.quantize(transformer, weights=qfloat8)

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Saving to {OUT_DIR} ...")
    qmodel.save_pretrained(OUT_DIR)

    del transformer, qmodel
    torch.cuda.empty_cache()
    print("Transformer done.\n")


def quantize_text_encoder():
    te_dir = f"{OUT_DIR}/text_encoder"
    marker = f"{te_dir}/model.safetensors"
    if os.path.exists(marker):
        print(f"[skip] text encoder already quantized -> {marker}")
        return

    print("Loading bf16 text encoder...")
    from transformers import Qwen3ForCausalLM

    text_encoder = Qwen3ForCausalLM.from_pretrained(
        f"{MODELS_PATH}/text_encoder", local_files_only=True
    ).to("cuda", torch.bfloat16)

    print(f"Peak VRAM before quantize: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    print("Quantizing text encoder to fp8...")
    quantize(text_encoder, weights=qfloat8)
    freeze(text_encoder)

    os.makedirs(te_dir, exist_ok=True)
    print(f"Saving to {te_dir} ...")
    # max_shard_size forced huge so this never shards into multiple files --
    # load_quantized_models() only ever looks for a single "model.safetensors".
    text_encoder.save_pretrained(te_dir, safe_serialization=True, max_shard_size="100GB")

    with open(f"{te_dir}/quanto_qmap.json", "w") as f:
        json.dump(quantization_map(text_encoder), f, indent=4)

    del text_encoder
    torch.cuda.empty_cache()
    print("Text encoder done.\n")


def copy_tokenizer():
    tok_dir = f"{OUT_DIR}/tokenizer"
    if os.path.isdir(tok_dir) and os.listdir(tok_dir):
        print(f"[skip] tokenizer already copied -> {tok_dir}")
        return

    print("Copying tokenizer (unquantized, nothing to do here)...")
    from transformers import Qwen2TokenizerFast

    tok = Qwen2TokenizerFast.from_pretrained(
        f"{MODELS_PATH}/tokenizer", local_files_only=True
    )
    tok.save_pretrained(tok_dir)
    print("Tokenizer done.\n")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    quantize_transformer()
    quantize_text_encoder()
    copy_tokenizer()

    print(f"\nAll done. fp8 checkpoint is at {OUT_DIR}")
    print("Now set in your config:")
    print(f'  "int8_models_path": "FLUX.2-klein-9B-fp8"')
    print('  "enable_int8_quantization": true')
    print("\nNote: VAE and scheduler are NOT part of this checkpoint -- they")
    print("always load from the original bf16 FLUX.2-klein-9B/{vae,scheduler}")
    print("regardless of the quantization flag, so nothing to do there.")
