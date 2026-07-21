#!/bin/bash
set -e

# Start ComfyUI in background
cd /comfyui
python3 main.py --listen 0.0.0.0 --port 8188 &
COMFY_PID=$!

# Wait for ComfyUI to be ready
echo "Waiting for ComfyUI to start..."
for i in {1..60}; do
    if curl -s http://localhost:8188/system_stats > /dev/null 2>&1; then
        echo "ComfyUI is ready!"
        break
    fi
    sleep 2
done

# If ComfyUI didn't start, exit with error
if ! curl -s http://localhost:8188/system_stats > /dev/null 2>&1; then
    echo "ERROR: ComfyUI failed to start"
    exit 1
fi

echo "ComfyUI started with PID $COMFY_PID"
