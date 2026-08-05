#!/bin/bash
set -e

echo "=== EntryPoint Wrapper: Completely overriding ComfyUI startup ==="
echo "Timestamp: $(date -Iseconds)"

# =============================================================================
# CRITICAL: sombi/base entrypoint runs ComfyUI on port 3000 via 'exec python main.py'
# which replaces this shell. We MUST bypass it entirely and run ComfyUI ourselves.
# =============================================================================

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
# VOLUME SETUP (must be done before ComfyUI starts)
# =============================================================================
echo ""
echo "=== Setting up network volume symlinks ==="

# Check if /runpod-volume is actually mounted
is_runpod_mounted() {
    if mount | grep -q "on /runpod-volume "; then
        return 0  # true - is mounted
    fi
    # Also check if /runpod-volume is a symlink to an existing directory
    if [ -L "/runpod-volume" ] && [ -d "/runpod-volume" ]; then
        local target=$(readlink -f /runpod-volume 2>/dev/null)
        if [ -n "$target" ] && [ "$target" != "/" ]; then
            return 0
        fi
    fi
    return 1  # false - not mounted or doesn't exist
}

# Setup /workspace based on whether network volume is available
if is_runpod_mounted; then
    echo "Network volume detected at /runpod-volume"
    mkdir -p /runpod-volume/workspace
    if [ ! -L /workspace ] || [ -d /workspace ]; then
        rm -rf /workspace
        ln -s /runpod-volume/workspace /workspace
        echo "Created /workspace -> /runpod-volume/workspace"
    fi

    mkdir -p /runpod-volume/comfyui
    if [ ! -e /comfyui ] || [ -L /comfyui ]; then
        [ -e /comfyui ] && rm -rf /comfyui
        ln -s /runpod-volume/comfyui /comfyui
        echo "Created /comfyui -> /runpod-volume/comfyui"
    fi

    # Ensure model directories exist on network volume
    mkdir -p /runpod-volume/models/{vae,diffusion_models,text_encoders,loras,latent_upscale_models,output,temp}
else
    echo "WARNING: /runpod-volume not mounted, using local storage"
    echo "This will use ephemeral storage which is limited (~5GB)"
fi

# =============================================================================
# START ComfyUI DIRECTLY (NOT via base image entrypoint)
# =============================================================================
echo ""
echo "=== Starting ComfyUI on port 8188 (bypassing base entrypoint) ==="

cd /comfyui

# Start ComfyUI directly with correct port
# --disable-auto-launch: Don't auto-launch browser
# --disable-metadata: Don't add metadata to outputs
# --verbose: Set log level
# --log-stdout: Log to stdout
# CRITICAL: --base-directory ensures ALL paths point to /comfyui (symlinked to network volume)
/opt/venv/bin/python main.py \
    --listen 0.0.0.0 \
    --port 8188 \
    --disable-auto-launch \
    --disable-metadata \
    --verbose "${COMFY_LOG_LEVEL:-INFO}" \
    --log-stdout \
    --base-directory /comfyui \
    --output-directory /comfyui/output \
    --temp-directory /comfyui/temp \
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
        echo "ComfyUI is ready at http://localhost:8188 (after $i attempts)"
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

cd /
exec /opt/venv/bin/python /handler.py
