# Dockerfile for ComfyUI LTX Video Serverless Worker
# Based on sombi/comfyui:base-torch2.8.0-cu124 with LTX Video customizations
# Models are loaded from network volume at /runpod-volume/models/

FROM sombi/comfyui:base-torch2.8.0-cu124

# =============================================================================
# Install git for cloning custom nodes
# =============================================================================
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Clone custom nodes for LTX Video
# =============================================================================
RUN git clone https://github.com/kijai/ComfyUI-KJNodes /comfyui/custom_nodes/ComfyUI-KJNodes && \
    cd /comfyui/custom_nodes/ComfyUI-KJNodes && git checkout main

RUN git clone https://github.com/Lightricks/ComfyUI-LTXVideo /comfyui/custom_nodes/ComfyUI-LTXVideo && \
    cd /comfyui/custom_nodes/ComfyUI-LTXVideo && git checkout master

RUN git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite /comfyui/custom_nodes/ComfyUI-VideoHelperSuite && \
    cd /comfyui/custom_nodes/ComfyUI-VideoHelperSuite && git checkout main

RUN git clone https://github.com/rgthree/rgthree-comfy /comfyui/custom_nodes/rgthree-comfy && \
    cd /comfyui/custom_nodes/rgthree-comfy && git checkout main

# =============================================================================
# CRITICAL: Redirect /workspace to network volume
# The sombi/base image does rsync of ComfyUI+venv to /workspace (~20GB+).
# Without this, /workspace fills ephemeral storage (~5GB) and crashes.
# This symlink must be created BEFORE the base image's entrypoint runs.
# =============================================================================
RUN rm -rf /workspace && \
    mkdir -p /runpod-volume/workspace && \
    ln -s /runpod-volume/workspace /workspace && \
    echo "Created symlink: /workspace -> /runpod-volume/workspace"

# =============================================================================
# Link models from network volume to ComfyUI models directory
# CRITICAL: Without these, ComfyUI defaults to /workspace/ComfyUI (ephemeral ~5GB)
# which will fill up and cause "No space left on device" errors.
# With --base-directory /comfyui and these symlinks, all writes go to network volume.
# =============================================================================
RUN mkdir -p /comfyui/models && \
    ln -sf /runpod-volume/models/vae /comfyui/models/vae && \
    ln -sf /runpod-volume/models/diffusion_models /comfyui/models/diffusion_models && \
    ln -sf /runpod-volume/models/text_encoders /comfyui/models/text_encoders && \
    ln -sf /runpod-volume/models/loras /comfyui/models/loras && \
    ln -sf /runpod-volume/models/latent_upscale_models /comfyui/models/latent_upscale_models && \
    ln -sf /runpod-volume/output /comfyui/output && \
    ln -sf /runpod-volume/temp /comfyui/temp

# =============================================================================
# CRITICAL: Mirror ComfyUI's full dependency set
# Note: /comfyui/requirements.txt may not exist in base image, skip if missing
# =============================================================================
RUN pip3 install --no-cache-dir \
    "transformers>=4.50.3,<5" "huggingface-hub<1.0" \
    && for r in /comfyui/custom_nodes/*/requirements.txt; do \
         [ -f "$r" ] && pip3 install --no-cache-dir -r "$r" || true; \
       done

# =============================================================================
# Install serverless handler dependencies
# =============================================================================
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt
RUN pip3 install --no-cache-dir opencv-python imageio_ffmpeg

# =============================================================================
# Copy handler files
# =============================================================================
COPY handler.py /handler.py
COPY api-workflow.json /api-workflow.json
COPY workflow.json /workflow.json

# Make scripts executable
RUN chmod +x /handler.py /start.sh

# =============================================================================
# Environment variables
# =============================================================================

# ComfyUI startup configuration
ENV COMFY_API_AVAILABLE_MAX_RETRIES=300
ENV COMFY_API_AVAILABLE_INTERVAL_MS=1000
ENV COMFY_LOG_LEVEL=INFO
ENV HISTORY_POLL_INTERVAL=2000
ENV HISTORY_TIMEOUT=600

# =============================================================================
# Override base image entrypoint with our start.sh
# =============================================================================
ENTRYPOINT ["/bin/bash", "/start.sh"]
