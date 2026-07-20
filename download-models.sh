#!/bin/bash
# Download LTX Video models to network volume
# Usage: HF_TOKEN=hf_xxx ./download-models.sh

set -e

# For RunPod serverless: network volume mounts at /runpod-volume
# ComfyUI automatically detects models under /runpod-volume/models/
MODELS_DIR="${VOLUME_MODELS_DIR:-/runpod-volume/models}"
HF_TOKEN="${HF_TOKEN:-}"

echo "=== Downloading LTX Video Models ==="
echo "Target: $MODELS_DIR"

# Create directories
mkdir -p "$MODELS_DIR/diffusion_models/sulpher"
mkdir -p "$MODELS_DIR/text_encoders/ltx2.3"
mkdir -p "$MODELS_DIR/vae/ltx2.3"
mkdir -p "$MODELS_DIR/latent_upscale_models"

# 1. Main Diffusion Model - LTX Video (~5GB)
echo "1. Downloading diffusion model (ltx2310eros_v1.safetensors ~5GB)..."
TARGET="$MODELS_DIR/diffusion_models/sulpher/ltx2310eros_v1.safetensors"
if [ -f "$TARGET" ] && [ $(stat -c%s "$TARGET") -gt 5000000000 ]; then
    echo "   Already exists, skipping..."
else
    wget -q --progress=bar:force \
        "https://huggingface.co/TenStrip/LTX2.3-10Eros/resolve/main/10Eros_v1-fp8mixed_learned.safetensors" \
        -O "$TARGET" \
        --header "Authorization: Bearer $HF_TOKEN"
fi
echo "   Done: $(du -sh $TARGET 2>/dev/null | cut -f1)"

# 2. Text Encoder - Gemma (~24GB)
echo "2. Downloading Gemma text encoder (~24GB)..."
TARGET="$MODELS_DIR/text_encoders/ltx2.3/gemma3-12B-hereticx-sikaworld.safetensors"
if [ -f "$TARGET" ] && [ $(stat -c%s "$TARGET") -gt 20000000000 ]; then
    echo "   Already exists, skipping..."
else
    wget -q --progress=bar:force \
        "https://huggingface.co/Sikaworld1990/gemma3-12B-hereticx-sikaworld-ltx-2/resolve/main/gemma3-12B-hereticx-sikaworld.safetensors" \
        -O "$TARGET" \
        --header "Authorization: Bearer $HF_TOKEN"
fi
echo "   Done: $(du -sh $TARGET 2>/dev/null | cut -f1)"

# 3. Text Projection (fix 0 bytes file)
echo "3. Fixing text projection..."
TARGET="$MODELS_DIR/text_encoders/ltx2.3/ltx-2.3_text_projection_bf16.safetensors"
rm -f "$TARGET"
wget -q --progress=bar:force \
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/text_encoders/ltx-2.3_text_projection_bf16.safetensors" \
    -O "$TARGET" \
    --header "Authorization: Bearer $HF_TOKEN"
echo "   Done: $(du -sh $TARGET 2>/dev/null | cut -f1)"

# 4. TAE VAE (required by Node 20 - VAELoader for preview override)
echo "4. Downloading TAE VAE (taeltx2_3.safetensors)..."
TARGET="$MODELS_DIR/vae/ltx2.3/taeltx2_3.safetensors"
if [ -f "$TARGET" ] && [ $(stat -c%s "$TARGET") -gt 20000000 ]; then
    echo "   Already exists, skipping..."
else
    wget -q --progress=bar:force \
        "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/taeltx2_3.safetensors" \
        -O "$TARGET" \
        --header "Authorization: Bearer $HF_TOKEN"
fi
echo "   Done: $(du -sh $TARGET 2>/dev/null | cut -f1)"

# 5. Spatial Upscaler x2 v1.1 (required by Node 9 - LatentUpscaleModelLoader)
echo "5. Downloading Spatial Upscaler x2 v1.1..."
TARGET="$MODELS_DIR/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
if [ -f "$TARGET" ] && [ $(stat -c%s "$TARGET") -gt 500000000 ]; then
    echo "   Already exists, skipping..."
else
    wget -q --progress=bar:force \
        "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors" \
        -O "$TARGET"
fi
echo "   Done: $(du -sh $TARGET 2>/dev/null | cut -f1)"

echo ""
echo "=== Download Complete ==="
echo "Diffusion: $(du -sh $MODELS_DIR/diffusion_models/sulpher/ 2>/dev/null | cut -f1)"
echo "Text Encoders: $(du -sh $MODELS_DIR/text_encoders/ltx2.3/ 2>/dev/null | cut -f1)"
