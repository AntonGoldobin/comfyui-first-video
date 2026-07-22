# Minimal Dockerfile for ComfyUI LTX Video Serverless
# Models are loaded from network volume at /runpod-volume/models/

FROM runpod/worker-comfyui:5.8.4-base

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install required custom nodes
RUN git clone https://github.com/kijai/ComfyUI-KJNodes /comfyui/custom_nodes/ComfyUI-KJNodes && \
    cd /comfyui/custom_nodes/ComfyUI-KJNodes && git checkout main

RUN git clone https://github.com/Lightricks/ComfyUI-LTXVideo /comfyui/custom_nodes/ComfyUI-LTXVideo && \
    cd /comfyui/custom_nodes/ComfyUI-LTXVideo && git checkout master

RUN git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite /comfyui/custom_nodes/ComfyUI-VideoHelperSuite && \
    cd /comfyui/custom_nodes/ComfyUI-VideoHelperSuite && git checkout main

RUN git clone https://github.com/rgthree/rgthree-comfy /comfyui/custom_nodes/rgthree-comfy && \
    cd /comfyui/custom_nodes/rgthree-comfy && git checkout main

# Create models directory and symlink model directories to network volume
RUN mkdir -p /comfyui/models && \
    ln -sf /runpod-volume/models/vae /comfyui/models/vae && \
    ln -sf /runpod-volume/models/diffusion_models /comfyui/models/diffusion_models && \
    ln -sf /runpod-volume/models/text_encoders /comfyui/models/text_encoders && \
    ln -sf /runpod-volume/models/loras /comfyui/models/loras && \
    ln -sf /runpod-volume/models/latent_upscale_models /comfyui/models/latent_upscale_models

# Install Python dependencies for serverless handler
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
RUN pip install opencv-python

# Copy handler files (overwrite base image's handler with our LTX Video handler)
COPY handler.py /handler.py
COPY runpod_handler.py /runpod_handler.py
COPY api-workflow.json /api-workflow.json
COPY workflow.json /workflow.json
COPY start.sh /start.sh

# Increase ComfyUI startup timeout (base image defaults are too short for network volumes)
# 300 seconds should be enough for ComfyUI to load all models from network volume
ENV COMFY_API_AVAILABLE_MAX_RETRIES=300
ENV COMFY_API_AVAILABLE_INTERVAL_MS=1000
