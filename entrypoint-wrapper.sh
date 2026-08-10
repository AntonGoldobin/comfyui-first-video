#!/bin/bash
set -e

echo "=== EntryPoint Wrapper: ComfyUI startup (image venv) ==="
echo "Timestamp: $(date -Iseconds)"

# =============================================================================
# Why this wrapper exists
# =============================================================================
# The sombi/base entrypoint runs ComfyUI on port 3000 via 'exec python main.py'
# which replaces this shell. We MUST bypass it entirely and run ComfyUI ourselves
# on port 8188, using the IMAGE's installed /opt/venv (not the network volume's
# /workspace/venv, which carries a stale CUDA-13-compiled torchaudio and
# triggers "libcudart.so.13: cannot open shared object" errors that block
# startup). See commit message for details.

# =============================================================================
# GPU PRE-FLIGHT CHECK
# =============================================================================
echo ""
echo "=== GPU Pre-flight Check ==="

if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found"
    exit 1
fi

GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1)
if [ $? -ne 0 ]; then
    echo "ERROR: nvidia-smi query failed: $GPU_INFO"
    exit 1
fi
echo "GPU: $GPU_INFO"

echo "Testing CUDA kernel access..."
CUDA_TEST=$(/opt/venv/bin/python -c "
import torch
if not torch.cuda.is_available():
    print('ERROR: CUDA not available')
    exit(1)
device = torch.device('cuda:0')
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
# NETWORK VOLUME SETUP — only for model + output dirs (not for venv or ComfyUI)
# =============================================================================
echo ""
echo "=== Verifying network volume mounts ==="

is_runpod_mounted() {
    if mount | grep -q "on /runpod-volume "; then
        return 0
    fi
    if [ -L "/runpod-volume" ] && [ -d "/runpod-volume" ]; then
        local target=$(readlink -f /runpod-volume 2>/dev/null)
        if [ -n "$target" ] && [ "$target" != "/" ]; then
            return 0
        fi
    fi
    return 1
}

if is_runpod_mounted; then
    echo "Network volume detected at /runpod-volume"
    mkdir -p /runpod-volume/models/{vae,diffusion_models,text_encoders,loras,latent_upscale_models,output,temp}

    # Ensure model symlinks exist in IMAGE's /comfyui (Dockerfile creates them at
    # build time, but verify + repair in case the image was rebuilt without them).
    mkdir -p /comfyui/models
    for d in vae diffusion_models text_encoders loras latent_upscale_models; do
        if [ ! -e /comfyui/models/$d ]; then
            ln -sf /runpod-volume/models/$d /comfyui/models/$d
        fi
    done
    mkdir -p /runpod-volume/output /runpod-volume/temp
    if [ ! -e /comfyui/output ]; then
        ln -sf /runpod-volume/output /comfyui/output
    fi
    if [ ! -e /comfyui/temp ]; then
        ln -sf /runpod-volume/temp /comfyui/temp
    fi
    echo "Model + output directories ready"
else
    echo "WARNING: /runpod-volume not mounted, using local /comfyui (ephemeral)"
fi

# =============================================================================
# Verify IMAGE's venv is what we want (not network volume's)
# =============================================================================
echo ""
echo "=== Verifying image venv ==="
if [ ! -x /opt/venv/bin/python ]; then
    echo "ERROR: /opt/venv/bin/python not found in image"
    exit 1
fi
VENV_TORCHAUDIO=$(/opt/venv/bin/python -c "import torchaudio; print(torchaudio.__version__)" 2>&1 || echo "IMPORT-FAILED")
echo "Image venv torchaudio: $VENV_TORCHAUDIO"
VENV_TORCH=$(/opt/venv/bin/python -c "import torch; print(torch.__version__, 'cuda', torch.version.cuda)" 2>&1)
echo "Image venv torch: $VENV_TORCH"

# =============================================================================
# START ComfyUI using IMAGE's /opt/venv and IMAGE's /comfyui
# =============================================================================
echo ""
echo "=== Starting ComfyUI on port 8188 (image venv, image code) ==="

cd /comfyui

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
echo $COMFY_PID > /tmp/comfyui.pid
echo "PID file written to /tmp/comfyui.pid"

# =============================================================================
# WAIT for ComfyUI to be ready (with verbose curl diagnostics — mirrors
# check_server() in handler.py since the handler is never called if startup
# fails, so this loop is the only window we have to see why)
# =============================================================================
echo ""
echo "=== Waiting for ComfyUI to be ready ==="

MAX_RETRIES=${COMFY_API_AVAILABLE_MAX_RETRIES:-300}
INTERVAL_MS=${COMFY_API_AVAILABLE_INTERVAL_MS:-1000}
INTERVAL_SEC=0.$(printf "%03d" $((INTERVAL_MS % 1000)))
INTERVAL_SEC=${INTERVAL_SEC%.*}

# Verbose diagnostic flags (Aug 10 2026 — Phase 15 close-out).
_LOGGED_FIRST_HTTP=0
_LOGGED_FIRST_LOG=0
_LOGGED_FIRST_DEATH=0
LAST_LOG_LINES=0

HTTP_CODE="000"
HTTP_BODY=""

for i in $(seq 1 $MAX_RETRIES); do
    # Check if process is still alive
    if ! kill -0 $COMFY_PID 2>/dev/null; then
        echo "ERROR: ComfyUI process died during startup. Logs:"
        cat /tmp/comfyui.log
        exit 1
    fi

    # Check HTTP endpoint with verbose diagnostic
    HTTP_CODE=$(curl -s -o /tmp/comfyui_probe.txt -w "%{http_code}" http://localhost:8188/system_stats 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        echo "ComfyUI is ready at http://localhost:8188 (after $i attempts)"
        break
    fi

    # DIAG: log first HTTP failure in detail
    if [ "$_LOGGED_FIRST_HTTP" -eq 0 ] && [ $i -ge 3 ]; then
        HTTP_BODY=$(cat /tmp/comfyui_probe.txt 2>/dev/null | head -c 200 || true)
        echo "[DIAG] First HTTP probe (attempt $i): code=$HTTP_CODE body=${HTTP_BODY:0:200}"
        _LOGGED_FIRST_HTTP=1
    fi

    # DIAG: log first 10 lines of ComfyUI log so we see where it's stuck
    if [ "$_LOGGED_FIRST_LOG" -eq 0 ] && [ -f /tmp/comfyui.log ]; then
        LOG_LINES=$(wc -l < /tmp/comfyui.log 2>/dev/null || echo 0)
        if [ "$LOG_LINES" -gt 5 ]; then
            echo "[DIAG] ComfyUI log has $LOG_LINES lines so far — last 10:"
            tail -10 /tmp/comfyui.log
            _LOGGED_FIRST_LOG=1
            LAST_LOG_LINES=$LOG_LINES
        fi
    fi

    # DIAG: every 60 attempts, show last 5 lines so we see if log is updating
    if [ $((i % 60)) -eq 0 ] && [ -f /tmp/comfyui.log ]; then
        CUR_LINES=$(wc -l < /tmp/comfyui.log 2>/dev/null || echo 0)
        if [ "$CUR_LINES" -ne "$LAST_LOG_LINES" ]; then
            echo "[DIAG] ComfyUI log progressed (was $LAST_LOG_LINES, now $CUR_LINES) — last 5:"
            tail -5 /tmp/comfyui.log
            LAST_LOG_LINES=$CUR_LINES
        else
            echo "[DIAG] ComfyUI log stuck at $CUR_LINES lines (likely hung)"
        fi
    fi

    if [ $((i % 10)) -eq 0 ]; then
        echo "Waiting for ComfyUI... ($i/$MAX_RETRIES)"
    fi

    if [ $i -lt 50 ]; then
        sleep 1
    else
        sleep $INTERVAL_SEC
    fi
done

# Final check
if [ "$HTTP_CODE" != "200" ]; then
    echo "ERROR: ComfyUI failed to start after $MAX_RETRIES attempts. Final HTTP=$HTTP_CODE. Logs:"
    cat /tmp/comfyui.log
    exit 1
fi

echo "ComfyUI is ready at http://localhost:8188"

# =============================================================================
# START RunPod Serverless Handler (image's venv)
# =============================================================================
echo ""
echo "=== Starting RunPod Serverless Handler ==="

cd /
exec /opt/venv/bin/python /handler.py
