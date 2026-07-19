# Minimal Dockerfile for ComfyUI LTX Video Serverless
# Only includes models needed for basic T2V / I2V workflow

FROM runpod/worker-comfyui:5.8.4-base

ARG HF_TOKEN=""

# Install required custom nodes
RUN git clone https://github.com/kijai/ComfyUI-KJNodes /comfyui/custom_nodes/ComfyUI-KJNodes && \
    cd /comfyui/custom_nodes/ComfyUI-KJNodes && \
    git checkout 17cddd4959f00182c66954be2a7a32a6481d81be 2>/dev/null || \
    (git fetch origin 17cddd4959f00182c66954be2a7a32a6481d81be --depth=1 && git checkout 17cddd4959f00182c66954be2a7a32a6481d81be) || \
    echo "WARN: KJNodes commit unreachable"

RUN git clone https://github.com/Lightricks/ComfyUI-LTXVideo /comfyui/custom_nodes/ComfyUI-LTXVideo && \
    cd /comfyui/custom_nodes/ComfyUI-LTXVideo && \
    git checkout 531512f7286963dc7aff1fd8bf5556e95eae03af 2>/dev/null || \
    (git fetch origin 531512f7286963dc7aff1fd8bf5556e95eae03af --depth=1 && git checkout 531512f7286963dc7aff1fd8bf5556e95eae03af) || \
    echo "WARN: LTXVideo commit unreachable"

RUN git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite /comfyui/custom_nodes/ComfyUI-VideoHelperSuite && \
    cd /comfyui/custom_nodes/ComfyUI-VideoHelperSuite && \
    git checkout 449839959f0153fb8a57211a9364c55163935ca9 2>/dev/null || \
    (git fetch origin 449839959f0153fb8a57211a9364c55163935ca9 --depth=1 && git checkout 449839959f0153fb8a57211a9364c55163935ca9) || \
    echo "WARN: VideoHelperSuite commit unreachable"

RUN git clone https://github.com/rgthree/rgthree-comfy /comfyui/custom_nodes/rgthree-comfy && \
    cd /comfyui/custom_nodes/rgthree-comfy && \
    git checkout 8ff50e4521881eca1fe26aec9615fc9362474931 2>/dev/null || \
    (git fetch origin 8ff50e4521881eca1fe26aec9615fc9362474931 --depth=1 && git checkout 8ff50e4521881eca1fe26aec9615fc9362474931) || \
    echo "WARN: rgthree commit unreachable"

# Download ONLY required models (minimal set for LTX Video)

# 1. VAE (required for decoding)
RUN HF_TOKEN=$HF_TOKEN comfy model download \
    --url 'https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_video_vae_bf16.safetensors' \
    --relative-path models/vae \
    --filename 'ltx2.3/LTX23_video_vae_bf16.safetensors'

# 2. Audio VAE (for audio generation if needed)
RUN HF_TOKEN=$HF_TOKEN comfy model download \
    --url 'https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_audio_vae_bf16.safetensors' \
    --relative-path models/vae \
    --filename 'ltx2.3/LTX23_audio_vae_bf16.safetensors'

# 3. Text Encoder - Gemma (required for LTX Video)
RUN HF_TOKEN=$HF_TOKEN comfy model download \
    --url 'https://huggingface.co/Sikaworld1990/gemma3-12B-hereticx-sikaworld-ltx-2/resolve/main/gemma3-12B-hereticx-sikaworld.safetensors' \
    --relative-path models/text_encoders \
    --filename 'ltx2.3/gemma3-12B-hereticx-sikaworld.safetensors'

# 4. Text Encoder Projection
RUN HF_TOKEN=$HF_TOKEN comfy model download \
    --url 'https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/text_encoders/ltx-2.3_text_projection_bf16.safetensors' \
    --relative-path models/text_encoders \
    --filename 'ltx2.3/ltx-2.3_text_projection_bf16.safetensors'

# 5. Main Diffusion Model - LTX Video (~5GB)
RUN HF_TOKEN=$HF_TOKEN comfy model download \
    --url 'https://huggingface.co/TenStrip/LTX2.3-10Eros/resolve/main/10Eros_v1-fp8mixed_learned.safetensors' \
    --relative-path models/diffusion_models \
    --filename 'sulpher/ltx2310eros_v1.safetensors'

# 6. LoRA (optional - remove if not used in workflow)
RUN HF_TOKEN=$HF_TOKEN comfy model download \
    --url 'https://huggingface.co/TenStrip/LTX2.3_Distilled_Lora_1.1_Experiments/resolve/main/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors' \
    --relative-path models/loras \
    --filename 'LTX23/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors'

# REMOVED (not needed for basic workflow):
# - Extra VAE models (taeltx2_3)
# - Latent upscale models
# - Spatial upscale models
# - Extra LoRA models
