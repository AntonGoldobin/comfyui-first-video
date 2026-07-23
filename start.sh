#!/bin/bash
set -e

echo "=== ComfyUI LTX Video Serverless Worker Starting ==="
echo "Timestamp: $(date -Iseconds)"

# =============================================================================
# GPU PRE-FLIGHT CHECK
# =============================================================================
echo ""
echo "=== GPU Pre-flight Check ==="

# Check nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found"
    exit 1
fi

# Query GPU info
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1)
if [ $? -ne 0 ]; then
    echo "ERROR: nvidia-smi query failed: $GPU_INFO"
    exit 1
fi
echo "GPU: $GPU_INFO"

# Actual CUDA kernel test - try to allocate and use GPU memory
echo "Testing CUDA kernel access..."
CUDA_TEST=$(python3 -c "
import torch
if not torch.cuda.is_available():
    print('ERROR: CUDA not available')
    exit(1)
device = torch.device('cuda:0')
# Try to allocate and use GPU
x = torch.randn(1024, 1024, device=device)
y = x @ x.T
result = y.cpu().numpy()
print(f'CUDA test passed: {result.shape}')
" 2>&1)
if [ $? -ne 0 ]; then
    echo "ERROR: CUDA kernel test failed: $CUDA_TEST"
    exit 1
fi
echo "CUDA test: $CUDA_TEST"

# =============================================================================
# START ComfyUI
# =============================================================================
echo ""
echo "=== Starting ComfyUI ==="

cd /comfyui

# Start ComfyUI with official runpod/worker-comfyui flags
# --disable-auto-launch: Don't auto-launch browser
# --disable-metadata: Don't add metadata to outputs
# --verbose: Set log level
# --log-stdout: Log to stdout
/opt/venv/bin/python main.py \
    --listen 0.0.0.0 \
    --port 8188 \
    --disable-auto-launch \
    --disable-metadata \
    --verbose "${COMFY_LOG_LEVEL:-INFO}" \
    --log-stdout \
    > /tmp/comfyui.log 2>&1 &

COMFY_PID=$!
echo "ComfyUI started with PID: $COMFY_PID"

# CRITICAL: Write PID to file for handler to monitor
echo $COMFY_PID > /tmp/comfyui.pid
echo "PID file written to /tmp/comfyui.pid"

# =============================================================================
# WAIT for ComfyUI to be ready
# =============================================================================
echo ""
echo "=== Waiting for ComfyUI to be ready ==="

MAX_RETRIES=${COMFY_API_AVAILABLE_MAX_RETRIES:-300}
INTERVAL_MS=${COMFY_API_AVAILABLE_INTERVAL_MS:-1000}
INTERVAL_SEC=0.$(printf "%03d" $((INTERVAL_MS % 1000)))
INTERVAL_SEC=${INTERVAL_SEC%.*}

# Wait for /system_stats endpoint
for i in $(seq 1 $MAX_RETRIES); do
    # Check if process is still alive
    if ! kill -0 $COMFY_PID 2>/dev/null; then
        echo "ERROR: ComfyUI process died during startup. Logs:"
        cat /tmp/comfyui.log
        exit 1
    fi

    # Check HTTP endpoint
    if curl -s -f http://localhost:8188/system_stats > /dev/null 2>&1; then
        echo "ComfyUI is ready! (after $i attempts)"
        break
    fi

    if [ $((i % 10)) -eq 0 ]; then
        echo "Waiting for ComfyUI... ($i/$MAX_RETRIES)"
    fi

    # Variable sleep - shorter initially, longer as we wait
    if [ $i -lt 50 ]; then
        sleep 1
    else
        sleep $INTERVAL_SEC
    fi
done

# Final check
if ! curl -s -f http://localhost:8188/system_stats > /dev/null 2>&1; then
    echo "ERROR: ComfyUI failed to start after $MAX_RETRIES attempts. Logs:"
    cat /tmp/comfyui.log
    exit 1
fi

echo "ComfyUI is ready at http://localhost:8188"

# =============================================================================
# START RunPod Serverless Handler
# =============================================================================
echo ""
echo "=== Starting RunPod Serverless Handler ==="

# Run using the correct Python venv
cd /
/opt/venv/bin/python /handler.py
