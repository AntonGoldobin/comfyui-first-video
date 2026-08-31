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
# Reconcile network-volume venv with what current ComfyUI needs.
#
# The network volume's /workspace/venv was rsync'd from a previous sombi
# image and is out of sync with the current /workspace/ComfyUI code:
#
#   - torchaudio wheel is cu130-built → loads libcudart.so.13 at import
#     and crashes. Must REMOVE.
#   - torchvision is missing entirely → ComfyUI imports it at top level
#     (comfy/ldm/cascade/stage_c_coder.py:19) and dies with
#     ModuleNotFoundError if absent. Must INSTALL.
#
# Install path: pytorch cu128 index, version pinned to match torch 2.8.0.
# This is slow on first run (~30-60s for torchvision wheel ~700 MB) but
# pip caches in $HOME/.cache/pip, so subsequent container starts are fast
# (or skip install entirely if torchvision is already present).
# =============================================================================
case "$PYTHON_BIN" in
    /workspace/venv*|/runpod-volume/workspace/venv*)
        echo ""
        echo "=== Reconciling network-volume venv (remove bad torchaudio, install torchvision) ==="

        # Step 1: remove cu130 torchaudio (causes libcudart.so.13 errors).
        rm -rf /workspace/venv/lib/python*/site-packages/torchaudio 2>/dev/null
        rm -rf /workspace/venv/lib/python*/site-packages/torchaudio-* 2>/dev/null
        rm -rf /runpod-volume/workspace/venv/lib/python*/site-packages/torchaudio 2>/dev/null
        rm -rf /runpod-volume/workspace/venv/lib/python*/site-packages/torchaudio-* 2>/dev/null
        echo "  - removed cu130 torchaudio"

        # Step 2: install torchvision if missing (matching torch 2.8.0+cu128).
        if "$PYTHON_BIN" -c "import torchvision; print('torchvision', torchvision.__version__)" 2>/dev/null; then
            echo "  - torchvision already present, skipping install"
        else
            echo "  - torchvision missing; installing torchvision==0.23.0+cu128 (matching torch 2.8.0+cu128)..."
            if "$PYTHON_BIN" -m pip install --quiet --no-cache-dir \
                torchvision==0.23.0+cu128 \
                --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -5; then
                NEW_TV=$("$PYTHON_BIN" -c "import torchvision; print(torchvision.__version__)" 2>&1)
                echo "  - torchvision installed: $NEW_TV"
            else
                echo "  - WARNING: torchvision install FAILED — ComfyUI may fail to import"
            fi
        fi

        echo "Done (audio modules will fail to import; ComfyUI has torchvision)"
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
    mkdir -p /runpod-volume/models/{vae,diffusion_models,text_encoders,loras,latent_upscale_models,checkpoints,output,temp}

    # Phase 28o: ComfyUI scans $COMFYUI_DIR/models (== /runpod-volume/workspace/ComfyUI/models
    # when --base-directory points there). The sombi base image rsyncs the network volume
    # with empty placeholder dirs (e.g. checkpoints/ contains only "put_checkpoints_here"),
    # which makes the [ ! -e ] check below a no-op — ComfyUI then sees empty checkpoint
    # lists. Always replace placeholder dirs with symlinks to /runpod-volume/models/<name>.
    mkdir -p "$COMFYUI_DIR/models"
    for d in vae diffusion_models text_encoders loras latent_upscale_models checkpoints; do
        TARGET="$COMFYUI_DIR/models/$d"
        # Replace: if it's a directory (real or placeholder) with no real checkpoints,
        # blow it away and symlink. If it's already a symlink, leave it.
        if [ -L "$TARGET" ]; then
            : # already a symlink, leave it
        elif [ -d "$TARGET" ]; then
            # If directory only contains placeholders (any file starting with "put_"
            # is a sombi-base convention marker like put_text_encoder_files_here),
            # treat as empty and replace. Otherwise leave alone (user-provided dir).
            REAL_FILES=$(ls -A "$TARGET" 2>/dev/null | grep -v '^put_' || true)
            if [ -z "$REAL_FILES" ]; then
                echo "  $TARGET is empty placeholder — replacing with symlink to /runpod-volume/models/$d"
                rm -rf "$TARGET"
                ln -sf /runpod-volume/models/$d "$TARGET"
            else
                echo "  $TARGET has real content — leaving in place"
            fi
        else
            ln -sf /runpod-volume/models/$d "$TARGET"
        fi
    done
    mkdir -p /runpod-volume/output /runpod-volume/temp
    if [ -L "$COMFYUI_DIR/output" ]; then
        : # already a symlink, leave it
    elif [ ! -e "$COMFYUI_DIR/output" ]; then
        ln -sf /runpod-volume/output "$COMFYUI_DIR/output"
    fi
    if [ -L "$COMFYUI_DIR/temp" ]; then
        : # already a symlink, leave it
    elif [ ! -e "$COMFYUI_DIR/temp" ]; then
        ln -sf /runpod-volume/temp "$COMFYUI_DIR/temp"
    fi

    # Phase 27: bootstrap LTX-2 19b distilled models if missing.
    # Idempotent — re-running a worker will skip files that already exist.
    # First cold start will download ~68 GB on a fresh network volume.
    # Re-disable via SKIP_DOWNLOAD=1 in endpoint env vars.
    if [ -f /usr/local/bin/download-models-ltx2.sh ]; then
        echo "=== Phase 27: ensure LTX-2 19b distilled models present ==="
        SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-}" HF_TOKEN="${HF_TOKEN:-}" \
            bash /usr/local/bin/download-models-ltx2.sh || \
            echo "⚠️  model download failed (will continue; ComfyUI will fail on missing models)"
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

# Phase 30: persist ComfyUI log to network volume so we can inspect it after a hang.
# Implementation: bash `tee` writes ComfyUI's stdout/stderr to BOTH /tmp/comfyui.log
# (local, fast) AND $LOG_FILE on network volume (persistent beyond worker lifetime).
# Original attempt passed "$LOG_FILE" as a second positional arg to --verbose, but
# ComfyUI's VerboseAction parsed it incorrectly and ComfyUI exited during startup,
# causing the pod to restart every ~135s. The tee approach avoids that parser.
LOG_FILE="/runpod-volume/comfyui-run-${RUNPOD_POD_ID:-unknown}.log"
echo "ComfyUI log: $LOG_FILE (also at /tmp/comfyui.log)"

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
    --use-sage-attention \
    --fp8_e4m3fn-unet \
    > /tmp/comfyui.log 2>&1 &

COMFY_PID=$!
echo "ComfyUI PID: $COMFY_PID"

# Mirror /tmp/comfyui.log to network volume in background. Uses tail -F to
# survive log rotations. Started AFTER ComfyUI launch so the file exists.
if [ -d /runpod-volume ] && [ ! -f "$LOG_FILE" ]; then
    touch "$LOG_FILE"
fi
if [ -f "$LOG_FILE" ] || [ -w /runpod-volume ]; then
    tail -F /tmp/comfyui.log >> "$LOG_FILE" 2>/dev/null &
    LOG_TAIL_PID=$!
    echo "Log mirror PID: $LOG_TAIL_PID → $LOG_FILE"
fi
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
# START RunPod Serverless Handler (unless --no-handler is set)
# =============================================================================
# Phase 30: --no-handler flag is for debug Pods (interactive ComfyUI testing).
# handler.py launches a RunPod serverless worker that exits immediately on a
# Pod (no RUNPOD_POD_ID env var wired to a serverless endpoint), which would
# restart the container and cut off browser access every ~30s. When set, we
# exec sleep infinity to keep the container alive so the user can interact
# with ComfyUI at port 8188 indefinitely.
#
# Triggered by RunPod Pod dockerArgs: "--no-handler"
if [ "${1:-}" = "--no-handler" ]; then
    echo ""
    echo "=== --no-handler flag set — ComfyUI is up, keeping pod alive ==="
    exec sleep infinity
fi

echo ""
echo "=== Starting RunPod Serverless Handler ==="
cd /
exec "$PYTHON_BIN" /handler.py
