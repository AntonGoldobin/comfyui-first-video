# Minimal Dockerfile for ComfyUI LTX Video Serverless
# Models are loaded from network volume at /runpod-volume/models/
# See download-models.sh for downloading models to volume

FROM runpod/worker-comfyui:5.8.4-base

ARG HF_TOKEN
# HF_TOKEN from environment (set by RunPod as secret) - no default!

# Install required custom nodes - no specific commits, use latest
RUN git clone https://github.com/kijai/ComfyUI-KJNodes /comfyui/custom_nodes/ComfyUI-KJNodes && \
    cd /comfyui/custom_nodes/ComfyUI-KJNodes && git checkout main

RUN git clone https://github.com/Lightricks/ComfyUI-LTXVideo /comfyui/custom_nodes/ComfyUI-LTXVideo && \
    cd /comfyui/custom_nodes/ComfyUI-LTXVideo && git checkout main

RUN git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite /comfyui/custom_nodes/ComfyUI-VideoHelperSuite && \
    cd /comfyui/custom_nodes/ComfyUI-VideoHelperSuite && git checkout main

RUN git clone https://github.com/rgthree/rgthree-comfy /comfyui/custom_nodes/rgthree-comfy && \
    cd /comfyui/custom_nodes/rgthree-comfy && git checkout main

# Symlink model directories to network volume
RUN ln -sf /runpod-volume/models/vae /comfyui/models/vae && \
    ln -sf /runpod-volume/models/diffusion_models /comfyui/models/diffusion_models && \
    ln -sf /runpod-volume/models/text_encoders /comfyui/models/text_encoders && \
    ln -sf /runpod-volume/models/loras /comfyui/models/loras && \
    ln -sf /runpod-volume/models/latent_upscale_models /comfyui/models/latent_upscale_models
