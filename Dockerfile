# Minimal Dockerfile for ComfyUI LTX Video Serverless
# Models are loaded from network volume at /runpod-volume/models/
# See download-models.sh for downloading models to volume

FROM runpod/worker-comfyui:5.8.4-base

# HF_TOKEN from environment (set by RunPod as secret) - no default!

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

# Symlink model directories to network volume
# Network volume mounts at /runpod-volume, models are at /runpod-volume/models/
# ComfyUI looks for models in /comfyui/models/ by default
RUN ln -sf /runpod-volume/models/vae /comfyui/models/vae && \
    ln -sf /runpod-volume/models/diffusion_models /comfyui/models/diffusion_models && \
    ln -sf /runpod-volume/models/text_encoders /comfyui/models/text_encoders && \
    ln -sf /runpod-volume/models/loras /comfyui/models/loras && \
    ln -sf /runpod-volume/models/latent_upscale_models /comfyui/models/latent_upscale_models
