#!/bin/bash
set -e

# Start ComfyUI in background
echo "Starting ComfyUI..."
cd /comfyui
python3 main.py --listen 0.0.0.0 --port 8188 --disable-smart-memory > /tmp/comfyui.log 2>&1 &
COMFY_PID=$!

# Wait for ComfyUI to be ready
echo "Waiting for ComfyUI to start..."
for i in {1..60}; do
    if curl -s http://localhost:8188/system_stats > /dev/null 2>&1; then
        echo "ComfyUI is ready!"
        break
    fi
    echo "Waiting... ($i/60)"
    sleep 2
done

if ! curl -s http://localhost:8188/system_stats > /dev/null 2>&1; then
    echo "ERROR: ComfyUI failed to start. Logs:"
    cat /tmp/comfyui.log
    exit 1
fi

echo "Starting RunPod serverless handler..."
# Start the runpod serverless handler
python3 -c "
import runpod
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import after ComfyUI is ready
from runpod_handler import _worker

async def handler(job):
    return await _worker.handler(job)

runpod.serverless.start({'handler': handler})
"
