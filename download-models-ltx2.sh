#!/bin/bash
# Download LTX-2 (19b distilled) models to network volume.
# Used by entrypoint-wrapper.sh at worker cold start to bootstrap a
# fresh network volume. Idempotent — re-running checks file size
# before each download.
#
# Models required by ltx2-t2v-distilled.json (LTX-2 19b distilled):
#   1. diffusion checkpoint: ltx-2-19b-distilled.safetensors           (~43 GB)
#   2. text encoder:         gemma-3-12b-it-qat-q4_0-unquantized/       (~24 GB)
#   3. latent upscaler:      ltx-2-spatial-upscaler-x2-1.0.safetensors  (~1 GB)
#   4. audio VAE:            same as diffusion checkpoint (combined)
#
# Folder layout (after symlinks in /comfyui/models/*):
#   /runpod-volume/models/checkpoints/ltx-2-19b-distilled.safetensors
#   /runpod-volume/models/text_encoders/gemma-3-12b-it-qat-q4_0-unquantized/
#   /runpod-volume/models/latent_upscale_models/ltx-2-spatial-upscaler-x2-1.0.safetensors
#
# Total: ~68 GB. Network volume f3falnf3r0 is 200 GB.
#
# Env vars:
#   HF_TOKEN: HuggingFace token for gated/private repos (none required here)
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

# 2. Gemma 3 12B IT QAT text encoder (multi-file, ~24 GB total)
# Download via huggingface_hub mirror — single command clones the
# subfolder only. The workflow expects model-00001-of-00005.safetensors
# so we need the whole directory.
if [ -z "${SKIP_DOWNLOAD:-}" ] || [ "$SKIP_DOWNLOAD" != "1" ]; then
    GEMMA_DIR="$MODELS_DIR/text_encoders/gemma-3-12b-it-qat-q4_0-unquantized"
    if [ -f "$GEMMA_DIR/model-00001-of-00005.safetensors" ] && \
       [ "$(stat -c%s "$GEMMA_DIR/model-00001-of-00005.safetensors" 2>/dev/null || stat -f%z "$GEMMA_DIR/model-00001-of-00005.safetensors" 2>/dev/null)" -gt 4000000000 ]; then
        echo "  ✓ Gemma 3 12B already present"
    else
        echo "  ⬇️  Gemma 3 12B IT QAT → $GEMMA_DIR"
        mkdir -p "$GEMMA_DIR"
        # Use python+httpx for parallel downloads since this is multi-file
        python3 - <<'PYEOF' || echo "  ⚠️  python download failed, retrying via wget"
import os, sys
from pathlib import Path
out = Path(os.environ.get("GEMMA_DIR", "/runpod-volume/models/text_encoders/gemma-3-12b-it-qat-q4_0-unquantized"))
repo = "google/gemma-3-12b-it-qat-q4_0-unquantized"
files = [
    "config.json", "generation_config.json", "model.safetensors.index.json",
    "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
    "model-00001-of-00005.safetensors",
    "model-00002-of-00005.safetensors",
    "model-00003-of-00005.safetensors",
    "model-00004-of-00005.safetensors",
    "model-00005-of-00005.safetensors",
]
out.mkdir(parents=True, exist_ok=True)
import urllib.request
hdr = {}
tok = os.environ.get("HF_TOKEN", "")
if tok:
    hdr["Authorization"] = f"Bearer {tok}"
for fn in files:
    target = out / fn
    if target.exists() and target.stat().st_size > 1_000_000:
        print(f"  ✓ {fn} already present")
        continue
    url = f"https://huggingface.co/{repo}/resolve/main/{fn}"
    print(f"  ⬇️  {fn}")
    req = urllib.request.Request(url, headers=hdr)
    with urllib.request.urlopen(req, timeout=600) as r, open(target, "wb") as f:
        while True:
            chunk = r.read(8 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    print(f"  ✓ {fn} done ({target.stat().st_size / 1e9:.2f} GB)")
PYEOF
    fi
fi

# 3. Spatial upscaler x2 v1.0 (~1 GB)
fetch "ltx-2-spatial-upscaler-x2-1.0" \
    "https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors" \
    "$MODELS_DIR/latent_upscale_models/ltx-2-spatial-upscaler-x2-1.0.safetensors" \
    500000000

echo ""
echo "=== Download complete ==="
echo "Total used: $(du -sh "$MODELS_DIR" 2>/dev/null | cut -f1)"
