#!/bin/bash
# Download LTX-2 (19b distilled) models to network volume.
# Used by entrypoint-wrapper.sh at worker cold start to bootstrap a
# fresh network volume. Idempotent — re-running checks file size
# before each download.
#
# Models required by ltx2-t2v-distilled.json (LTX-2 19b distilled):
#   1. diffusion checkpoint: ltx-2-19b-distilled.safetensors           (~43 GB)
#   2. text encoder:         Lightricks/LTX-2/text_encoder/            (~50 GB, Gemma 12B)
#   3. text projection:      Lightricks/LTX-2/connectors/              (~3 GB)
#   4. latent upscaler:      ltx-2-spatial-upscaler-x2-1.0.safetensors  (~1 GB)
#
# Folder layout (after symlinks in /comfyui/models/*):
#   /runpod-volume/models/checkpoints/ltx-2-19b-distilled.safetensors
#   /runpod-volume/models/text_encoders/Lightricks-LTX-2-text_encoder/
#   /runpod-volume/models/text_encoders/Lightricks-LTX-2-connectors/
#   /runpod-volume/models/latent_upscale_models/ltx-2-spatial-upscaler-x2-1.0.safetensors
#
# Total: ~97 GB. Network volume f3falnf3r0 is 200 GB.
#
# Why Lightricks/LTX-2 subdirs instead of google/gemma-3-12b-it-qat-q4_0-unquantized:
#   - google/gemma-3-12b-it-qat-q4_0-unquantized is a gated repo requiring license acceptance
#   - Lightricks redistributes Gemma weights themselves (non-gated, same weights)
#   - connectors/diffusion_pytorch_model.safetensors = text projection (Gemma → 768-dim)
#
# Env vars:
#   HF_TOKEN: HuggingFace token (recommended — anonymous downloads may rate-limit)
#   SKIP_DOWNLOAD=1: skip downloads (for testing/scripts)
#   MODELS_DIR: override target dir (default /runpod-volume/models)

set -e

MODELS_DIR="${MODELS_DIR:-/runpod-volume/models}"
HF_TOKEN="${HF_TOKEN:-}"

echo "=== LTX-2 19b Distilled model downloader ==="
echo "Target: $MODELS_DIR"
echo "Skip: ${SKIP_DOWNLOAD:-0}"

mkdir -p "$MODELS_DIR/checkpoints"
mkdir -p "$MODELS_DIR/text_encoders"
mkdir -p "$MODELS_DIR/latent_upscale_models"

# Helper: download only if file missing or too small
fetch() {
    local label="$1"
    local url="$2"
    local target="$3"
    local min_size="$4"

    if [ -n "${SKIP_DOWNLOAD:-}" ] && [ "$SKIP_DOWNLOAD" = "1" ]; then
        echo "[skip] $label (SKIP_DOWNLOAD=1)"
        return 0
    fi

    if [ -f "$target" ] && [ "$(stat -c%s "$target" 2>/dev/null || stat -f%z "$target" 2>/dev/null)" -gt "$min_size" ]; then
        echo "  ✓ $label already present ($(du -sh "$target" | cut -f1))"
        return 0
    fi

    echo "  ⬇️  $label → $target"
    mkdir -p "$(dirname "$target")"
    if [ -n "$HF_TOKEN" ]; then
        wget -q --progress=bar:force --header "Authorization: Bearer $HF_TOKEN" -O "$target" "$url"
    else
        wget -q --progress=bar:force -O "$target" "$url"
    fi
    echo "  ✓ $label done ($(du -sh "$target" | cut -f1))"
}

# 1. LTX-2 19b distilled checkpoint (~43 GB)
fetch "ltx-2-19b-distilled checkpoint" \
    "https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled.safetensors" \
    "$MODELS_DIR/checkpoints/ltx-2-19b-distilled.safetensors" \
    40000000000

# 2. Gemma 3 12B IT text encoder (~50 GB total, from Lightricks/LTX-2/text_encoder/)
# Lightricks redistributes Gemma 12B themselves at this path — same weights as
# the gated google/gemma-3-12b-it-qat-q4_0-unquantized but non-gated (no license
# acceptance required). Files include diffusion_pytorch_model-*.safetensors +
# model.safetensors.index.json + config.json + tokenizer/ side files.
# Tokenizer is in a separate HF subdir (Lightricks/LTX-2/tokenizer/) but
# CLIPLoader(type=ltxv) expects it alongside the model — download both into
# the same local dir so they're discoverable.
if [ -z "${SKIP_DOWNLOAD:-}" ] || [ "$SKIP_DOWNLOAD" != "1" ]; then
    GEMMA_DIR="$MODELS_DIR/text_encoders/Lightricks-LTX-2-text_encoder"
    if [ -f "$GEMMA_DIR/model.safetensors.index.json" ] && \
       [ "$(stat -c%s "$GEMMA_DIR/diffusion_pytorch_model-00001-of-00012.safetensors" 2>/dev/null || stat -f%z "$GEMMA_DIR/diffusion_pytorch_model-00001-of-00012.safetensors" 2>/dev/null)" -gt 1000000000 ]; then
        echo "  ✓ Gemma 3 12B (Lightricks mirror) already present"
    else
        echo "  �️  Gemma 3 12B (from Lightricks/LTX-2/text_encoder + tokenizer) → $GEMMA_DIR"
        mkdir -p "$GEMMA_DIR"
        GEMMA_DIR="$GEMMA_DIR" python3 - <<'PYEOF' || echo "  �️  python download failed, retrying via wget"
import os, sys
from pathlib import Path
out = Path(os.environ["GEMMA_DIR"])
hdr = {}
tok = os.environ.get("HF_TOKEN", "")
if tok:
    hdr["Authorization"] = f"Bearer {tok}"

# (HF subdir in repo, local filename) pairs — tokenizer goes into same local dir as text_encoder
all_files = [
    ("text_encoder", "config.json"),
    ("text_encoder", "generation_config.json"),
    ("text_encoder", "model.safetensors.index.json"),
    # NOTE: special_tokens_map.json is NOT in text_encoder/ on Lightricks/LTX-2
    # (it lives in tokenizer/). Skip to avoid HTTP 404 aborting the whole run.
    ("text_encoder", "diffusion_pytorch_model-00001-of-00012.safetensors"),
    ("text_encoder", "diffusion_pytorch_model-00002-of-00012.safetensors"),
    ("text_encoder", "diffusion_pytorch_model-00003-of-00012.safetensors"),
    ("text_encoder", "diffusion_pytorch_model-00004-of-00012.safetensors"),
    ("text_encoder", "diffusion_pytorch_model-00005-of-00012.safetensors"),
    ("text_encoder", "diffusion_pytorch_model-00006-of-00012.safetensors"),
    ("text_encoder", "diffusion_pytorch_model-00007-of-00012.safetensors"),
    ("text_encoder", "diffusion_pytorch_model-00008-of-00012.safetensors"),
    ("text_encoder", "diffusion_pytorch_model-00009-of-00012.safetensors"),
    ("text_encoder", "diffusion_pytorch_model-00010-of-00012.safetensors"),
    ("text_encoder", "diffusion_pytorch_model-00011-of-00012.safetensors"),
    ("text_encoder", "diffusion_pytorch_model-00012-of-00012.safetensors"),
    ("tokenizer", "added_tokens.json"),
    ("tokenizer", "chat_template.jinja"),
    ("tokenizer", "preprocessor_config.json"),
    ("tokenizer", "processor_config.json"),
    ("tokenizer", "special_tokens_map.json"),
    ("tokenizer", "tokenizer.json"),
    ("tokenizer", "tokenizer.model"),
    ("tokenizer", "tokenizer_config.json"),
]
out.mkdir(parents=True, exist_ok=True)
import urllib.request
for subdir, fn in all_files:
    target = out / fn
    if target.exists() and target.stat().st_size > 1_000:
        print(f"  ✓ {fn} already present")
        continue
    url = f"https://huggingface.co/Lightricks/LTX-2/resolve/main/{subdir}/{fn}"
    print(f"  ⬇️  {fn}")
    req = urllib.request.Request(url, headers=hdr)
    try:
        with urllib.request.urlopen(req, timeout=1200) as r, open(target, "wb") as f:
            while True:
                chunk = r.read(8 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        size_gb = target.stat().st_size / 1e9
        print(f"  ✓ {fn} done ({size_gb:.2f} GB)")
    except Exception as e:
        print(f"  ✗ {fn} FAILED: {e} (continuing with next file)")
        if target.exists():
            target.unlink()
        continue
PYEOF
    fi
fi

# 2b. Text projection connector (~3 GB, from Lightricks/LTX-2/connectors/)
# Maps Gemma 4096-dim hidden states to model's expected conditioning dim.
# ComfyUI loads via CLIPLoader(type=ltxv) or as part of ModelSamplingLTXV.
if [ -z "${SKIP_DOWNLOAD:-}" ] || [ "$SKIP_DOWNLOAD" != "1" ]; then
    CONN_DIR="$MODELS_DIR/text_encoders/Lightricks-LTX-2-connectors"
    fetch "Lightricks LTX-2 connector (text projection)" \
        "https://huggingface.co/Lightricks/LTX-2/resolve/main/connectors/diffusion_pytorch_model.safetensors" \
        "$CONN_DIR/diffusion_pytorch_model.safetensors" \
        2500000000
fi

# 3. Spatial upscaler x2 v1.0 (~1 GB)
fetch "ltx-2-spatial-upscaler-x2-1.0" \
    "https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors" \
    "$MODELS_DIR/latent_upscale_models/ltx-2-spatial-upscaler-x2-1.0.safetensors" \
    500000000

echo ""
echo "=== Download complete ==="
echo "Total used: $(du -sh "$MODELS_DIR" 2>/dev/null | cut -f1)"
