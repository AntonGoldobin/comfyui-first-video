# Dockerfile for ComfyUI LTX Video Serverless Worker
# Based on runpod/worker-comfyui:5.8.4-base with LTX Video customizations
# Models are loaded from network volume at /runpod-volume/models/

FROM runpod/worker-comfyui:5.8.4-base

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
# Link models from network volume to ComfyUI models directory
# =============================================================================
RUN mkdir -p /comfyui/models && \
    ln -sf /runpod-volume/models/vae /comfyui/models/vae && \
    ln -sf /runpod-volume/models/diffusion_models /comfyui/models/diffusion_models && \
    ln -sf /runpod-volume/models/text_encoders /comfyui/models/text_encoders && \
    ln -sf /runpod-volume/models/loras /comfyui/models/loras && \
    ln -sf /runpod-volume/models/latent_upscale_models /comfyui/models/latent_upscale_models

# =============================================================================
# CRITICAL: Mirror ComfyUI's full dependency set into /opt/venv
# This prevents missing module errors during workflow execution
# =============================================================================
RUN /opt/venv/bin/pip install --no-cache-dir \
    -r /comfyui/requirements.txt \
    && for r in /comfyui/custom_nodes/*/requirements.txt; do \
         [ -f "$r" ] && /opt/venv/bin/pip install --no-cache-dir -r "$r" || true; \
       done \
    && /opt/venv/bin/pip install --no-cache-dir "transformers>=4.50.3,<5" "huggingface-hub<1.0"

# =============================================================================
# Install serverless handler dependencies
# =============================================================================
COPY requirements.txt /tmp/requirements.txt
RUN /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt
RUN /opt/venv/bin/pip install --no-cache-dir opencv-python imageio_ffmpeg

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
