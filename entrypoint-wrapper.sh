#!/bin/bash
# NOTE: set +e (NOT set -e) — we want graceful fallback through candidate paths.
# Previous version's set -e plus missing /opt/venv/bin/python returned exit 127
# before any of our error logging ran. RunPod's worker-exit logs don't capture
# stderr from the ENTRYPOINT script reliably, so the only way to debug is to
# preflight-print the diagnostic information we need.
set +e

echo "=== EntryPoint Wrapper: ComfyUI startup (defensive) ==="
echo "Timestamp: $(date -Iseconds)"

# =============================================================================
# DIAGNOSTIC: List what exists so we know the image layout
# =============================================================================
echo ""
echo "=== DIAG: Available python/venv paths ==="
for p in /opt/venv/bin/python /workspace/venv/bin/python /comfyui/venv/bin/python /usr/local/bin/python3 /usr/bin/python3; do
    if [ -x "$p" ] 2>/dev/null; then
        echo "  [EXEC] $p"
    elif [ -f "$p" ] 2>/dev/null; then
        echo "  [FILE, NO EXEC] $p"
    else
        echo "  [MISSING] $p"
    fi
done

echo ""
echo "=== DIAG: Available ComfyUI directories ==="
for d in /comfyui /workspace/ComfyUI /runpod-volume/workspace/ComfyUI; do
    if [ -f "$d/main.py" ]; then
        echo "  [OK] $d/main.py"
    else
        echo "  [MISSING] $d/main.py"
    fi
done

# =============================================================================
# AUTO-DETECT PYTHON (must have torch + CUDA)
# =============================================================================
echo ""
echo "=== Auto-detecting Python with torch + CUDA ==="

PYTHON_BIN=""
for candidate in /opt/venv/bin/python /workspace/venv/bin/python /comfyui/venv/bin/python; do
    if [ -x "$candidate" ]; then
        if "$candidate" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            PYTHON_BIN="$candidate"
            echo "Selected (image/venv): $candidate"
            break
        else
            echo "  $candidate — torch/CUDA test FAILED (skipping)"
        fi
    fi
done

if [ -z "$PYTHON_BIN" ] && command -v python3 >/dev/null 2>&1; then
    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        PYTHON_BIN="$(command -v python3)"
        echo "Selected (system python3): $PYTHON_BIN"
    fi
fi

if [ -z "$PYTHON_BIN" ]; then
    echo "FATAL: no python with torch+CUDA found in any expected location"
    echo ""
    echo "ls /opt:"
    ls -la /opt 2>&1 | head -10
    echo ""
    echo "ls /workspace:"
    ls -la /workspace 2>&1 | head -10
    echo ""
    echo "ls /comfyui:"
    ls -la /comfyui 2>&1 | head -10
    echo ""
    echo "which python3:"
    which python3 2>&1
    echo ""
    echo "PATH: $PATH"
    exit 127
fi

# =============================================================================
# AUTO-DETECT COMFYUI DIR
# =============================================================================
echo ""
echo "=== Auto-detecting ComfyUI ==="

COMFYUI_DIR=""
for candidate in /comfyui /workspace/ComfyUI /runpod-volume/workspace/ComfyUI; do
    if [ -f "$candidate/main.py" ]; then
        COMFYUI_DIR="$candidate"
        echo "Selected: $candidate"
        break
    fi
done

if [ -z "$COMFYUI_DIR" ]; then
    echo "FATAL: no ComfyUI main.py found in /comfyui, /workspace/ComfyUI, /runpod-volume/workspace/ComfyUI"
    exit 127
fi

# =============================================================================
# SHOW VENV VERSIONS
# =============================================================================
echo ""
echo "=== Venv versions ==="
TORCHAUDIO_VER=$("$PYTHON_BIN" -c "import torchaudio; print(torchaudio.__version__)" 2>&1)
echo "torchaudio: $TORCHAUDIO_VER"
TORCH_VER=$("$PYTHON_BIN" -c "import torch; print(torch.__version__, 'cuda', torch.version.cuda)" 2>&1)
echo "torch: $TORCH_VER"

# =============================================================================
# If using network-volume venv, REMOVE bad CUDA-13 torchaudio
# (network volume's venv was rsync'd from a previous container with cu130
# torchaudio wheel; the image's runtime cuda is cu128, so loading its
# _torchaudio.so extension triggers libcudart.so.13 errors and blocks
# ComfyUI startup. Audio modules fail to import — soft error, ComfyUI
# continues.)
#
# CRITICAL: do NOT remove torchvision — ComfyUI's
# comfy/ldm/cascade/stage_c_coder.py imports torchvision at top-level and
# the worker dies with ModuleNotFoundError if we remove it. Only the
# torchaudio extension binary is incompatible with the cu128 runtime.
# =============================================================================
case "$PYTHON_BIN" in
    /workspace/venv*|/runpod-volume/workspace/venv*)
        echo ""
        echo "=== Removing stale CUDA-13 torchaudio from network-volume venv (keep torchvision) ==="
        rm -rf /workspace/venv/lib/python*/site-packages/torchaudio 2>/dev/null
        rm -rf /workspace/venv/lib/python*/site-packages/torchaudio-* 2>/dev/null
        rm -rf /runpod-volume/workspace/venv/lib/python*/site-packages/torchaudio 2>/dev/null
        rm -rf /runpod-volume/workspace/venv/lib/python*/site-packages/torchaudio-* 2>/dev/null
        echo "Done (audio modules will fail to import; ComfyUI continues)"
        ;;
    *)
        echo "Using image venv — bad torchaudio not present (no cleanup needed)"
        ;;
esac

# =============================================================================
# GPU PRE-FLIGHT CHECK
# =============================================================================
echo ""
echo "=== GPU Pre-flight Check ==="

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "FATAL: nvidia-smi not found"
    exit 1
fi

GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1)
echo "GPU: $GPU_INFO"

CUDA_TEST=$("$PYTHON_BIN" -c "
import torch
if not torch.cuda.is_available():
    print('ERROR: CUDA not available')
    exit(1)
x = torch.randn(1024, 1024, device='cuda:0')
y = x @ x.T
print(f'CUDA test passed: {y.cpu().numpy().shape}')
" 2>&1)
if [ $? -ne 0 ]; then
    echo "FATAL: CUDA kernel test failed: $CUDA_TEST"
    exit 1
fi
echo "CUDA: $CUDA_TEST"

# =============================================================================
# NETWORK VOLUME — only for model + output dirs (not for venv or ComfyUI)
# =============================================================================
echo ""
echo "=== Network volume setup ==="

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
    echo "Network volume mounted at /runpod-volume"
    mkdir -p /runpod-volume/models/{vae,diffusion_models,text_encoders,loras,latent_upscale_models,output,temp}

    mkdir -p "$COMFYUI_DIR/models"
    for d in vae diffusion_models text_encoders loras latent_upscale_models; do
        if [ ! -e "$COMFYUI_DIR/models/$d" ]; then
            ln -sf /runpod-volume/models/$d "$COMFYUI_DIR/models/$d"
        fi
    done
    mkdir -p /runpod-volume/output /runpod-volume/temp
    if [ ! -e "$COMFYUI_DIR/output" ]; then
        ln -sf /runpod-volume/output "$COMFYUI_DIR/output"
    fi
    if [ ! -e "$COMFYUI_DIR/temp" ]; then
        ln -sf /runpod-volume/temp "$COMFYUI_DIR/temp"
    fi
    echo "Model + output dirs ready"
else
    echo "WARNING: /runpod-volume not mounted, using local dirs (ephemeral)"
fi

# =============================================================================
# START COMFYUI
# =============================================================================
echo ""
echo "=== Starting ComfyUI on port 8188 ==="
echo "Python: $PYTHON_BIN"
echo "ComfyUI: $COMFYUI_DIR"
echo "Args: --listen 0.0.0.0 --port 8188 --disable-auto-launch --disable-metadata --base-directory $COMFYUI_DIR"

cd "$COMFYUI_DIR"

# shellcheck disable=SC2086
"$PYTHON_BIN" main.py \
    --listen 0.0.0.0 \
    --port 8188 \
    --disable-auto-launch \
    --disable-metadata \
    --verbose "${COMFY_LOG_LEVEL:-INFO}" \
    --log-stdout \
    --base-directory "$COMFYUI_DIR" \
    --output-directory "$COMFYUI_DIR/output" \
    --temp-directory "$COMFYUI_DIR/temp" \
    > /tmp/comfyui.log 2>&1 &

COMFY_PID=$!
echo "ComfyUI PID: $COMFY_PID"
echo $COMFY_PID > /tmp/comfyui.pid

# =============================================================================
# POLL for /system_stats — REAL polling (sleep 1 every iter)
#
# Previous bug: after i=50 the loop did `sleep 0.000` (computed from
# INTERVAL_SEC=0.0), so the loop actually spun for ~50 seconds, not
# $MAX_RETRIES seconds. Now MAX_RETRIES=600 → ~10 minutes of real polling.
# =============================================================================
echo ""
echo "=== Polling for /system_stats (real wait, no early sleep-0) ==="
MAX_RETRIES=${COMFY_API_AVAILABLE_MAX_RETRIES:-600}
LOGGED_FIRST_HTTP=0
LOGGED_FIRST_LOG=0
LAST_LOG_LINES=0
HTTP_CODE="000"

for i in $(seq 1 $MAX_RETRIES); do
    if ! kill -0 $COMFY_PID 2>/dev/null; then
        echo ""
        echo "ERROR: ComfyUI process $COMFY_PID died during startup"
        echo ""
        echo "=== FULL ComfyUI log ($(wc -l < /tmp/comfyui.log 2>/dev/null || echo 0) lines) ==="
        cat /tmp/comfyui.log
        exit 1
    fi

    HTTP_CODE=$(curl -s -o /tmp/comfyui_probe.txt -w "%{http_code}" --max-time 2 http://localhost:8188/system_stats 2>/dev/null || echo "000")

    if [ "$HTTP_CODE" = "200" ]; then
        echo "ComfyUI ready (after $i attempts)"
        break
    fi

    # DIAG: First HTTP failure with response body
    if [ "$LOGGED_FIRST_HTTP" -eq 0 ] && [ $i -ge 3 ]; then
        HTTP_BODY=$(cat /tmp/comfyui_probe.txt 2>/dev/null | head -c 300 || true)
        echo "[DIAG] attempt $i: HTTP=$HTTP_CODE body=${HTTP_BODY}"
        LOGGED_FIRST_HTTP=1
    fi

    # DIAG: First chunk of ComfyUI log so we see where it's stuck
    if [ "$LOGGED_FIRST_LOG" -eq 0 ] && [ -f /tmp/comfyui.log ]; then
        LOG_LINES=$(wc -l < /tmp/comfyui.log 2>/dev/null || echo 0)
        if [ "$LOG_LINES" -gt 5 ]; then
            echo "[DIAG] ComfyUI log first 20 lines (out of $LOG_LINES):"
            head -20 /tmp/comfyui.log
            LOGGED_FIRST_LOG=1
            LAST_LOG_LINES=$LOG_LINES
        fi
    fi

    # DIAG: every 60 attempts, show progress (or stuck)
    if [ $((i % 60)) -eq 0 ] && [ -f /tmp/comfyui.log ]; then
        CUR_LINES=$(wc -l < /tmp/comfyui.log 2>/dev/null || echo 0)
        if [ "$CUR_LINES" -ne "$LAST_LOG_LINES" ]; then
            echo "[DIAG] attempt $i — log progressing ($LAST_LOG_LINES → $CUR_LINES lines)"
            tail -3 /tmp/comfyui.log
            LAST_LOG_LINES=$CUR_LINES
        else
            echo "[DIAG] attempt $i — log STUCK at $CUR_LINES lines (ComfyUI may be hung)"
        fi
    fi

    if [ $((i % 30)) -eq 0 ]; then
        echo "Waiting for ComfyUI... ($i/$MAX_RETRIES, http=$HTTP_CODE)"
    fi

    sleep 1
done

if [ "$HTTP_CODE" != "200" ]; then
    echo ""
    echo "ERROR: ComfyUI failed to bind 8188 after $MAX_RETRIES attempts (final HTTP=$HTTP_CODE)"
    echo ""
    echo "=== FULL ComfyUI log ($(wc -l < /tmp/comfyui.log 2>/dev/null || echo 0) lines) ==="
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
exec "$PYTHON_BIN" /handler.py
