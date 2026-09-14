"""
modal_comfyui_minimax_h3.py — Modal.com deployment of ComfyUI for MiniMax-H3.

A SEPARATE Modal app from comfyui-ltx-video so:
  - LTX-2 endpoint stays untouched (no risk to the working production pipeline)
  - H3 models live on their own volume (don't share / fill LTX-2's volume)
  - Independent scaling (H3 is 22B params — slower cold-start, different profile)

Reuses the proven scaffold from modal_comfyui.py:
  - sombi base image (sombi/comfyui:base-torch2.8.0-cu124)
  - /ComfyUI mount, --output-directory /modal-data/output, ASGI proxy
  - copy_dir_contents from Modal Volume to /ComfyUI/models/
  - same 600s startup, 1800s timeout, 5s scaledown_window (R.117, 2026-09-01)

Differences from modal_comfyui.py:
  - app name: comfyui-minimax-h3
  - volume:   comfyui-minimax-h3-models (separate from comfyui-models)
  - H3-specific custom nodes: ComfyUI-Easy-Use (for easy loadImageBase64 input)
  - H3 models (~30 GB total): 22B UNet + Qwen3-VL CLIP + dual VAE (video + audio)
    + turbo LoRA

URL: https://anton722451--comfyui-minimax-h3-serve.modal.run
Env: MODAL_COMFYUI_BASE_URL_MINIMAX_H3 → above URL

Deploy:
  cd /Volumes/SSDNSKIY/VSCODE/comfyui-first-video
  modal deploy modal_comfyui_minimax_h3.py
  modal run modal_comfyui_minimax_h3.py::setup_minimax_h3_models --hf-token hf_xxx
"""

import os
import time
import threading
import logging
import modal
# FastAPI imports are deferred to method scope (Task #91) because modal
# CLI introspects this file locally during `modal deploy`, where fastapi
# may not be installed. The Modal container itself has fastapi via the
# sombi base image.

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("modal-comfyui-h3")

app = modal.App("comfyui-minimax-h3")
# SEPARATE volume — H3 (~30 GB) does NOT share with LTX-2's comfyui-models volume.
# This keeps LTX-2 deployment unaffected even if H3 fills the volume.
h3_models_volume = modal.Volume.from_name(
    "comfyui-minimax-h3-models", create_if_missing=True
)

# =============================================================================
# R.119 (2026-09-05): idempotency state for H3Generator.generate().
# Atomic nonce-keyed claim — see handoff doc + memory card
# r119-cold-start-idempotency-required-2026-09-05. Without this, cold-start
# retries create duplicate GPU jobs because the Modal Web Function gateway
# 307-redirects at 150s AFTER the function has already entered on the GPU
# container (verified empirically 2026-09-05).
# =============================================================================
# Task #218 (2026-09-13): DROP modal.Dict "jobs" — replaced by Modal-native
# FunctionCall.from_id(call_id).get(timeout=0) for async result retrieval.
# Why switch away from Dict (full RCA: research-modal-webhook-api-2026-09-13.md
# + proposal-fix-modal-303-cold-start-2026-09-13.md):
#   1. Dict eventual-consistency on container scaledown causes poll-7
#      "not_found" misclassification (4 incidents: adb39bb8, e36f6c07,
#      27d31af5, 24936235 — see task23 cluster).
#   2. The 150s gateway 303 transport break CANNOT be patched at the Dict
#      layer — only .spawn() (returns immediately, no blocking request thread)
#      bypasses the cold-start window entirely.
# FunctionCall is the community-blessed pattern per Modal docs:
#   https://modal.com/docs/guide/webhook-timeouts (see .spawn() section)
#   https://modal.com/docs/guide/trigger-deployed-functions (FunctionCall.from_id)
# Idempotency: worker keeps `call_id` for the lifetime of the job. If a
# worker retry happens, it re-polls the SAME call_id (not a new spawn), so
# concurrent submits with the same nonce are the WORKER's responsibility, not
# Modal's. Task #163 already added this dedup.
_R119_SECRETS = []

image = (
    modal.Image.from_registry("sombi/comfyui:base-torch2.8.0-cu124")
    .run_commands(
        # Install git + python3 (sombi base uses uv-managed Python, not /usr/bin/python3)
        "apt-get update && apt-get install -y --no-install-recommends git wget ca-certificates python3 python3-venv python3-pip && rm -rf /var/lib/apt/lists/*",
        "python3 -m pip install --no-cache-dir --break-system-packages pip setuptools wheel || true",
    )
    .run_commands(
        # REPLACE sombi's frozen ComfyUI v0.18.1 with a fresh clone of v0.33.2.
        # Why: v0.18.1 lacks the comfy.ldm.minimax module that provides the local
        # MiniMaxH3ImageToVideo / MiniMaxH3ReferenceToVideo nodes. We need v0.30+
        # (we pin v0.33.2 = latest stable as of 2026-08-20) for native H3 support.
        "rm -rf /ComfyUI",
        "git clone --depth=1 --branch v0.33.2 https://github.com/comfyanonymous/ComfyUI /ComfyUI",
    )
    .run_commands(
        # Install fresh ComfyUI's base requirements (replaces sombi's frozen /venv pkgs
        # with whatever v0.33.2 needs).
        "pip install --no-cache-dir -r /ComfyUI/requirements.txt",
    )
    .run_commands(
        # Custom nodes for H3:
        # - ComfyUI-Easy-Use: provides easy loadImageBase64 (node 220 in our workflow)
        # - ComfyUI-KJNodes: PatchSageAttentionKJ (recommended for H3 speedup, per
        #   https://docs.comfy.org/tutorials/video/minimax/minimax-h3) + ImageResizeKJv2
        # - ComfyUI-LTXVideo: MiniMaxH3SigmaShift (H3 sigma-shift helper node)
        # - Comfyui_Minimax_h3_latent_Upscaler (LBH-123-AI): MinimaxH3LatentUpscaler3D
        #   class for 24-channel H3 latent 3D upscaling. Used by
        #   minimax-h3-preview-upscaled style — first-pass 448x768 then 2x upscale
        #   to 896x1536 (~50-60% baseline cost, visually competitive). Weights are
        #   downloaded separately by setup_minimax_h3_models() into
        #   /ComfyUI/models/latent_upscale_models/. See MEMORY [[modal-h3-upscaler-2026-09-01]].
        "rm -rf /ComfyUI/custom_nodes/ComfyUI-Easy-Use /ComfyUI/custom_nodes/ComfyUI-KJNodes /ComfyUI/custom_nodes/ComfyUI-LTXVideo",
        "git clone --depth=1 https://github.com/yolain/ComfyUI-Easy-Use /ComfyUI/custom_nodes/ComfyUI-Easy-Use",
        "git clone --depth=1 https://github.com/kijai/ComfyUI-KJNodes /ComfyUI/custom_nodes/ComfyUI-KJNodes",
        "git clone --depth=1 https://github.com/Lightricks/ComfyUI-LTXVideo /ComfyUI/custom_nodes/ComfyUI-LTXVideo",
        "git clone --depth=1 https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler /ComfyUI/custom_nodes/Comfyui_Minimax_h3_latent_Upscaler",
    )
    .run_commands(
        # Install custom node deps
        "for r in /ComfyUI/custom_nodes/*/requirements.txt; do "
        "[ -f \"$r\" ] && pip install --no-cache-dir -r \"$r\" || true; "
        "done",
        "pip install --no-cache-dir opencv-python imageio_ffmpeg",
        # ASGI proxy deps (serve() uses FastAPI + httpx)
        "pip install --no-cache-dir fastapi httpx 'starlette>=0.36'",
        # Sage attention — soft requirement for MiniMax H3 (recommended in docs.comfy.org).
        # MUST install from git — SageAttention 2.2.0 is NOT on PyPI (latest = 1.0.6, Nov 2024).
        # PyPI install of "sageattention==2.2.0" silently fails with "No matching distribution",
        # which previously got masked by `|| true` and a Modal layer cache hit → deploy
        # "succeeded" but SA2 kernels never landed in the image. Now we install from git
        # main branch (pinned to commit d1a57a5 = 2.2.0 with sm90 FP8 kernels).
        #
        # CRITICAL: TORCH_CUDA_ARCH_LIST must be set BEFORE pip install. Modal build
        # workers have NO GPU → torch.utils.cpp_extension can't auto-detect compute
        # capability → build dies with "No target compute capabilities". H100 = Hopper
        # = sm_90; we add "9.0+PTX" for forward-compat.
        #
        # --no-build-isolation makes pip use the image's already-installed torch 2.8.0 /
        # triton 3.0+ / CUDA 12.4 (so the compiled kernels bind to ABI-compatible torch).
        # Without --no-build-isolation pip builds in an isolated env with OLD torch →
        # missing _qattn_sm90 / _qattn_sm120 kernels → "sageattention is not new enough"
        # or "could not determine CUDA architecture" errors at runtime.
        #
        # We keep fallback (warn-only) because SA2 is an optimization, not a hard
        # requirement — but we now log install output instead of `|| true`-silencing it.
        "TORCH_CUDA_ARCH_LIST='9.0+PTX' pip install --no-cache-dir --break-system-packages "
        "--no-build-isolation "
        "git+https://github.com/thu-ml/SageAttention.git 2>&1 | tail -10 "
        "|| echo 'WARN: sageattention install failed — falling back to comfy kitchen attention'",
        # Pin transformers/torchaudio/huggingface_hub to CUDA 12-compatible versions
        # (sombi base ships v5.x which conflicts with comfy.ldm.minimax imports).
        "pip install --no-cache-dir --break-system-packages transformers==4.56.0 huggingface_hub==0.36.2 torchaudio==2.8.0",
        # Patch MiniMax H3 2D upscaler to handle NestedTensor inputs (R.109 fix).
        # Symptom: AttributeError 'NestedTensor' object has no attribute 'clone'
        # at the line `s = latent["samples"].clone()` inside MinimaxH3LatentUpscalerNode2D.run().
        # Root cause: H3 sampler output (node 151 → SamplerCustomAdvanced) is wrapped
        # in torch.nested.nested_tensor.NestedTensor (SDPA / attention backend may emit
        # nested tensors under certain conditions). NestedTensor lacks .clone().
        # Fix: convert NestedTensor → dense padded tensor before clone.
        "python3 - <<'EOF'\nimport sys\np = '/ComfyUI/custom_nodes/Comfyui_Minimax_h3_latent_Upscaler/nodes/minimax_h3_latent_upscaler_2d.py'\ntry:\n    with open(p) as f:\n        src = f.read()\nexcept FileNotFoundError:\n    print(f'WARN: {p} not found — skipping upscaler patch', file=sys.stderr)\n    sys.exit(0)\nold = '        s = latent[\"samples\"].clone()\\n        orig_dtype = s.dtype'\nnew = '''        samples_in = latent[\"samples\"]\n        # Patch 2026-09-01 (R.109): NestedTensor inputs lack .clone(). Convert to dense.\n        if hasattr(samples_in, 'is_nested') and samples_in.is_nested:\n            samples_in = samples_in.to_padded_tensor(0)\n        s = samples_in.clone()\n        orig_dtype = s.dtype'''\nif old in src:\n    src = src.replace(old, new)\n    with open(p, 'w') as f:\n        f.write(src)\n    print('Patched: MinimaxH3LatentUpscalerNode2D NestedTensor → dense')\nelse:\n    print('PATCH ALREADY APPLIED or pattern not found — leaving file unchanged')\nEOF",
    )
    .entrypoint([])  # disable base image entrypoint; we start ComfyUI ourselves
)


# =============================================================================
# Setup job: download H3 models to Modal Volume
# =============================================================================
@app.function(
    image=image,
    volumes={"/modal-data": h3_models_volume},
    cpu=4,
    memory=8192,
    timeout=7200,
    startup_timeout=600,
)
def setup_minimax_h3_models(hf_token: str = "") -> dict:
    """Download MiniMax-H3 models to Modal Volume. Idempotent.

    Total ~30 GB. Files confirmed against https://huggingface.co/Comfy-Org/MiniMax-H3:
      - diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors (~14 GB)
        (fl2va = first-last-frame-to-video. Pair with the 8-step turbo LoRA below.)
      - text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors (~15 GB)
      - vae/minimax_h3_video_vae_fp16.safetensors  (NOT fp8 — fp8 doesn't exist)
      - vae/minimax_h3_audio_vae_fp32.safetensors  (NOT fp8 — fp8 doesn't exist)
      - loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors
        (8-step is more flexible than 4-step; supports both 5s and 10s outputs.)
    """
    os.environ["HF_TOKEN"] = hf_token
    import urllib.request
    hdr = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    # Symlink /runpod-volume → /modal-data so any custom-node code that uses
    # /runpod-volume paths works.
    if os.path.isdir("/runpod-volume") and not os.path.islink("/runpod-volume"):
        os.system("rm -rf /runpod-volume && ln -s /modal-data /runpod-volume")

    # ---- H3 model registry — VERIFIED against Comfy-Org/MiniMax-H3 2026-08-20 ----
    H3_FILES = [
        # (relative_path_under_models/, source_url, expected_min_bytes)
        (
            "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            14_000_000_000,
        ),
        (
            "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            15_000_000_000,
        ),
        (
            "vae/minimax_h3_video_vae_fp16.safetensors",
            "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors",
            500_000_000,
        ),
        (
            "vae/minimax_h3_audio_vae_fp32.safetensors",
            "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors",
            200_000_000,
        ),
        (
            "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
            "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
            100_000_000,
        ),
        # Mystic XXX — community style LoRA for H3 (lynaNSFW/mysticxxx_MM_H3, V4 pruned)
        # strength_model 0.5-0.9; stacks after Turbo. See lynaNSFW HF repo for trigger guidance.
        (
            "loras/MysticXXX_MMH3-V4.safetensors",
            "https://huggingface.co/lynaNSFW/mysticxxx_MM_H3/resolve/main/MysticXXX_MMH3-V4.safetensors",
            100_000_000,
        ),
        # H3 3D latent upscaler (bf16) — required by minimax-h3-preview-upscaled style.
        # BCTHW-aware: handles H3's 24-channel latent including temporal dim. Default
        # 2x scale (40% of training data). Disk: ~691 MB.
        # Placed in ComfyUI/models/latent_upscale_models/ where the LBH custom node's
        # COMBO model_name field auto-discovers. See workflow node 500 in
        # minimax-h3-preview-upscaled.cleaned.json + MEMORY [[modal-h3-upscaler-2026-09-01]].
        (
            "latent_upscale_models/minimax_h3_latent_upscaler_3d_bf16.safetensors",
            "https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/resolve/main/minimax_h3_latent_upscaler_3d_bf16.safetensors",
            600_000_000,  # 691 MB - 10% tolerance
        ),
    ]

    log.info("=== Downloading MiniMax-H3 model set (~30 GB) ===")
    for rel_path, url, expected_min in H3_FILES:
        dst = f"/modal-data/models/{rel_path}"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst) and os.path.getsize(dst) > expected_min * 0.95:
            log.info(f"{rel_path}: already present ({os.path.getsize(dst)/1e9:.2f} GB)")
            continue
        log.info(f"Downloading {rel_path} from {url}")
        req = urllib.request.Request(url, headers=hdr)
        try:
            with urllib.request.urlopen(req, timeout=7200) as r, open(dst, "wb") as f:
                while True:
                    chunk = r.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            log.info(f"{rel_path}: done ({os.path.getsize(dst)/1e9:.2f} GB)")
        except Exception as e:
            log.error(f"Failed to download {rel_path}: {e}")
            raise

    h3_models_volume.commit()
    return {"status": "ok", "files": len(H3_FILES)}


# =============================================================================
# Task #218 (2026-09-13): serve() /api/status endpoint DELETED — replaced by
# Modal-native FunctionCall.from_id(call_id).get(timeout=0) pattern (Option A
# per Modal docs). The serve() function existed to host a Dict-based status
# poll endpoint for worker polling, but that architecture had two failure modes:
#   1. Cold-start 303 transport break (150s gateway limit)
#   2. Dict eventual-consistency on container scaledown
# Both are eliminated by switching api_run to .spawn() and adding api_result
# below (H3Generator class method).
# Worker's MODAL_STATUS_URL env var now points to H3Generator's api_result URL
# (auto-named by Modal as `<app>-<class>-<method>.modal.run`). Same env var
# name retained for back-compat; URL pattern changes from `/?nonce=` to
# `/?call_id=` (query-param style because @modal.fastapi_endpoint does NOT
# support path args — see [[modal-fastapi-endpoint-no-path-arg-2026-09-10]]).
# =============================================================================


# =============================================================================
# R.119 (2026-09-05) + Task #218 (2026-09-13): H3Generator — direct method-based
# invocation, no HTTP roundtrip. Returns video bytes directly so worker never
# touches /api/view (which serves from ComfyUI's in-memory OUTPUTS_MAP and
# 404s after scaledown).
#
# Idempotency model (v4 — Task #218 / Option A):
#   R.119 v1-v3 used modal.Dict("jobs") for nonce-keyed atomic claim +
#   result caching. REPLACED by Modal-native FunctionCall.from_id pattern.
#   Worker keeps `call_id` from spawn response; retries by re-polling
#   same call_id (not re-spawning). Concurrent submits with same nonce are
#   the WORKER's responsibility, not Modal's (Task #163 already added this).
#   See [[modal-fastapi-endpoint-no-path-arg-2026-09-10]] for URL pattern.
# Flow:
#   1. Run workflow via local ComfyUI on :8188 (same container, no HTTP proxy).
#   2. Explicit volume.commit() before returning bytes — replaces R.137 watcher.
# =============================================================================
@app.cls(
    image=image,
    volumes={"/modal-data": h3_models_volume},
    secrets=_R119_SECRETS,
    cpu=4,
    memory=16384,
    timeout=1800,                  # class-level container lifetime cap
    enable_memory_snapshot=True,   # snapshot after @modal.enter() completes
    min_containers=0,
    scaledown_window=5,
    max_containers=20,
    # 2026-09-09: buffer_containers REMOVED (was 1). Same rationale as serve().
    # MEMORY [[modal-buffer-removed-permanently-2026-09-09]].
    buffer_containers=0,
    startup_timeout=600,
    # COST-OPT (2026-09-08): us-only-east. See serve() decorator for rationale.
    # Rollback: add "us-west" back if "capacity exhausted in us-east" errors
    # return. MEMORY [[r119-modal-cost-physical-cause-2026-09-08]].
    region="us-east",
    gpu="H100",
)
class H3Generator:
    @modal.enter(snap=True)
    def setup(self):
        """Initialize GPU container once per cold start. Symlinks models,
        launches ComfyUI subprocess on :8188, waits for ready. Snapshotted."""
        import subprocess
        import shutil
        import pathlib
        import re

        log.info("=== H3Generator.setup() — initializing GPU container ===")

        # Symlink so any internal /runpod-volume paths in custom nodes work
        if os.path.isdir("/runpod-volume") and not os.path.islink("/runpod-volume"):
            os.system("rm -rf /runpod-volume && ln -s /modal-data /runpod-volume")

        # Find python interpreter
        python_bin = None
        for cand in [
            "/venv/bin/python3", "/venv/bin/python",
            "/opt/venv/bin/python", "/usr/bin/python3",
            "/usr/local/bin/python3",
        ]:
            if os.path.exists(cand) and os.access(cand, os.X_OK):
                python_bin = cand
                break
        if not python_bin:
            python_bin = shutil.which("python3") or shutil.which("python")
        if not python_bin:
            raise RuntimeError("No python interpreter found in container")
        log.info(f"Using python: {python_bin}")

        # Sync model copy (R.131: async copy was unreliable)
        def copy_dir_contents(src_dir, dst_dir):
            if not os.path.isdir(src_dir):
                return
            os.makedirs(dst_dir, exist_ok=True)
            for fname in os.listdir(dst_dir):
                p = f"{dst_dir}/{fname}"
                if os.path.islink(p):
                    try:
                        if os.path.realpath(p).startswith("/modal-data"):
                            os.unlink(p)
                    except Exception:
                        pass
            for fname in os.listdir(src_dir):
                src = f"{src_dir}/{fname}"
                dst = f"{dst_dir}/{fname}"
                if os.path.isfile(dst) and not os.path.islink(dst):
                    continue
                if os.path.islink(dst):
                    continue
                if os.path.isdir(src):
                    copy_dir_contents(src, dst)
                    continue
                log.info(f"Copying {src} → {dst}")
                shutil.copy2(src, dst)

        all_subs = (
            "diffusion_models", "text_encoders", "vae",
            "loras", "latent_upscale_models", "checkpoints",
        )
        for sub in all_subs:
            os.makedirs(f"/modal-data/models/{sub}", exist_ok=True)
            os.makedirs(f"/ComfyUI/models/{sub}", exist_ok=True)

        log.info("Sync model copy (~50 GB / ~3 min)")
        # ponytail: cold-start RCA — verify whether copy loop re-runs inside setup()
        # despite @modal.enter(snap=True). Snapshots capture FS state from this point,
        # so the loop SHOULD be skipped on warm starts. If INSTR_COPY END appears on
        # warm starts, copy is being re-executed (bottleneck confirmed).
        copy_start_ts = time.monotonic()
        copy_start_wall = time.time()
        log.info(f"[INSTR_COPY] model copy START wall={copy_start_wall:.3f} monotonic={copy_start_ts:.3f}")
        for sub in all_subs:
            copy_dir_contents(f"/modal-data/models/{sub}", f"/ComfyUI/models/{sub}")
        copy_duration = time.monotonic() - copy_start_ts
        log.info(f"[INSTR_COPY] model copy END wall={time.time():.3f} duration={copy_duration:.1f}s")
        log.info("Model copy complete")

        # Persist output to Modal Volume
        os.makedirs("/modal-data/output", exist_ok=True)
        if os.path.islink("/ComfyUI/output") or os.path.isdir("/ComfyUI/output"):
            if os.path.islink("/ComfyUI/output"):
                os.unlink("/ComfyUI/output")
            else:
                shutil.rmtree("/ComfyUI/output")
        os.symlink("/modal-data/output", "/ComfyUI/output")
        log.info("Linked /ComfyUI/output → /modal-data/output")

        # ------------------------------------------------------------------
        # Task #216 (2026-09-13): defer eager CUDA init in comfy.model_management.
        # Root cause: `main.py:239` imports `comfy.model_management`, which at
        # module-load runs `total_vram = get_total_memory(get_torch_device())`.
        # `get_torch_device()` calls `torch.cuda.current_device()` →
        # `torch._C._cuda_init()`. After a Modal @modal.enter(snap=True)
        # snapshot restore, GPU memory state is NOT in the snapshot (Modal
        # docs explicit). On the first import post-restore, the GPU may not
        # be bound yet → RuntimeError("No CUDA GPUs are available"). ComfyUI
        # crashes BEFORE its HTTP server starts → setup()'s /system_stats
        # poll loops 600× → RuntimeError("ComfyUI startup timeout").
        #
        # Fix: wrap the eager init in try/except (RuntimeError, KeyError,
        # TypeError, AttributeError). On failure, `total_vram = 0`. On
        # Linux (Modal runs Linux), `total_vram` is only read in a
        # Windows-only VRAM-reservation branch; the only Linux effect is a
        # slightly different startup log line.
        #
        # Task #216d (2026-09-14): broadened from `except RuntimeError` to
        # `except (RuntimeError, KeyError, TypeError, AttributeError)`. The
        # original wrap at this consumer site missed KeyError raised
        # INSIDE `get_total_memory` at line ~403 on
        # `stats['reserved_bytes.all.current']` when GPU telemetry dict is
        # missing fields during cold-start snap-restore (Modal app
        # `comfyui-minimax-h3` crashlooping since v48 / 2026-09-13 10:24,
        # last known-good v47 = `4ef858e`). TypeError/AttributeError also
        # covered defensively for adjacent dict-access failures
        # (None stats, stale device handle, etc.).
        #
        # Idempotent: sentinel comment check skips re-runs. Re-runs on a
        # new ComfyUI version that preserves the line shape will re-patch
        # automatically. setup() runs every cold-start, so image rebuilds
        # are not required.
        # ------------------------------------------------------------------
        _mm_path = pathlib.Path("/ComfyUI/comfy/model_management.py")
        _SENTINEL = "# H3_TASK216_PATCH: deferred_cuda_init"
        try:
            _mm_src = _mm_path.read_text()
            if _SENTINEL in _mm_src:
                log.info("model_management.py: H3_TASK216 patch already applied — skipping")
            else:
                _pattern = re.compile(
                    r"^(\s*)total_vram\s*=\s*get_total_memory\(get_torch_device\(\)\)\s*/\s*\(1024\s*\*\s*1024\)\s*$",
                    re.MULTILINE,
                )
                _m = _pattern.search(_mm_src)
                if _m:
                    _indent = _m.group(1)
                    _replacement = (
                        f"{_indent}# H3_TASK216_PATCH: deferred_cuda_init\n"
                        f"{_indent}try:\n"
                        f"{_indent}    total_vram = get_total_memory(get_torch_device()) / (1024 * 1024)\n"
                        f"{_indent}except (RuntimeError, KeyError, TypeError, AttributeError) as _e3_init_err:\n"
                        f"{_indent}    # Modal @modal.enter(snap=True) snapshots CPU+FS but NOT GPU state.\n"
                        f"{_indent}    # First import post-restore can race the GPU bind. Defer to 0;\n"
                        f"{_indent}    # total_vram is only read in Windows-only VRAM-reservation logic,\n"
                        f"{_indent}    # so on Linux this only affects the startup log line.\n"
                        f"{_indent}    # Task #216d (2026-09-14): broadened except tuple — get_total_memory\n"
                        f"{_indent}    # at line ~403 crashes with KeyError on stats['reserved_bytes.all.current']\n"
                        f"{_indent}    # when GPU telemetry dict is missing fields during cold-start snap-restore.\n"
                        f"{_indent}    # TypeError/AttributeError also covered defensively for adjacent dict-access\n"
                        f"{_indent}    # failures (None stats, stale device handle, etc.).\n"
                        f"{_indent}    total_vram = 0\n"
                        f"{_indent}    logging.warning(\n"
                        f"{_indent}        \"H3_TASK216d: deferred init in comfy.model_management caught %s: %s\",\n"
                        f"{_indent}        type(_e3_init_err).__name__, _e3_init_err,\n"
                        f"{_indent}    )"
                    )
                    _new_src = _mm_src[:_m.start()] + _replacement + _mm_src[_m.end():]
                    _mm_path.write_text(_new_src)
                    log.info("model_management.py: applied H3_TASK216 deferred-CUDA-init patch")
                else:
                    log.warning(
                        "model_management.py: H3_TASK216 pattern not found — "
                        "ComfyUI version may have changed; leaving file unchanged"
                    )
        except Exception as _patch_err:
            log.warning(f"model_management.py: H3_TASK216 patcher failed: {_patch_err}")

        # ------------------------------------------------------------------
        # Task #216b (2026-09-13): defer eager CUDA init in
        # get_torch_device() itself. The first patcher covered the
        # `total_vram` lookup at line 363, but ComfyUI's startup reaches
        # `get_torch_device()` EARLIER — `cuda_malloc_warning()` at
        # main.py:286 calls `get_torch_device()` at module load, which
        # runs `torch.cuda.current_device()` at the line below BEFORE the
        # patched line 363 ever runs. After a Modal snapshot restore, the
        # GPU is not yet bound → RuntimeError("No CUDA GPUs are
        # available") → ComfyUI crashes before HTTP server starts.
        #
        # Fix: wrap the `torch.cuda.current_device()` call inside the
        # else-branch fallback of get_torch_device() in try/except
        # RuntimeError. On failure, fall back to torch.device("cpu").
        # cpu_state will then take the CPU branch on the next call, but
        # at least import succeeds and the HTTP server starts.
        #
        # Idempotent: same sentinel-skipped re-run pattern as Task #216.
        # ------------------------------------------------------------------
        _SENTINEL_2 = "# H3_TASK216_PATCH_GET_TORCH_DEVICE: deferred_cuda_init"
        try:
            _mm_src = _mm_path.read_text()
            if _SENTINEL_2 in _mm_src:
                log.info("model_management.py: H3_TASK216b patch already applied — skipping")
            else:
                # Match the bare CUDA branch in get_torch_device()'s else-fallback.
                # Anchored on the leading 12-space indent (inside `else:` of the
                # function body) so we don't accidentally match a similar line
                # elsewhere. The line is unique in model_management.py.
                _pattern2 = re.compile(
                    r"^( {12})return torch\.device\(torch\.cuda\.current_device\(\)\)\s*$",
                    re.MULTILINE,
                )
                _m2 = _pattern2.search(_mm_src)
                if _m2:
                    _indent2 = _m2.group(1)
                    _replacement2 = (
                        f"{_indent2}# H3_TASK216_PATCH_GET_TORCH_DEVICE: deferred_cuda_init\n"
                        f"{_indent2}try:\n"
                        f"{_indent2}    return torch.device(torch.cuda.current_device())\n"
                        f"{_indent2}except RuntimeError as _e212_init_err:\n"
                        f"{_indent2}    # Modal @modal.enter(snap=True) snapshots CPU+FS but NOT GPU state.\n"
                        f"{_indent2}    # get_torch_device() is called at module load by cuda_malloc_warning(),\n"
                        f"{_indent2}    # BEFORE the previously-patched line 363 ever runs. Fall back to CPU\n"
                        f"{_indent2}    # so ComfyUI import succeeds and the HTTP server can start; the next\n"
                        f"{_indent2}    # get_torch_device() call (after snap restore completes GPU bind) will\n"
                        f"{_indent2}    # return the real GPU device normally.\n"
                        f"{_indent2}    logging.warning(\n"
                        f"{_indent2}        \"H3_TASK216b: deferred CUDA init in get_torch_device(): %s\",\n"
                        f"{_indent2}        _e212_init_err,\n"
                        f"{_indent2}    )\n"
                        f"{_indent2}    return torch.device(\"cpu\")"
                    )
                    _new_src = _mm_src[:_m2.start()] + _replacement2 + _mm_src[_m2.end():]
                    _mm_path.write_text(_new_src)
                    log.info("model_management.py: applied H3_TASK216b deferred-CUDA-init patch (get_torch_device line 212)")
                else:
                    log.warning(
                        "model_management.py: H3_TASK216b pattern not found — "
                        "ComfyUI version may have changed; leaving file unchanged"
                    )
        except Exception as _patch_err2:
            log.warning(f"model_management.py: H3_TASK216b patcher failed: {_patch_err2}")

        # ------------------------------------------------------------------
        # Task #216c (2026-09-13): fundamental fix — monkey-patch torch.cuda
        # at module top of comfy/model_management.py so ALL current and
        # future torch.cuda.* call sites in this module become tolerant.
        # Whack-a-mole line patching (Tasks #216, #216b) is incomplete —
        # Probe D2 found a THIRD eager CUDA-init site at model_management.py
        # line 1973 in should_use_bf16 (calls torch.cuda.get_device_properties
        # at UNETLoader load time, AFTER patches above fired). Task #216d,
        # #216e would be inevitable if we kept patching line-by-line.
        #
        # Fix: inject a monkey-patch block right after `from __future__ import
        # annotations` that wraps torch.cuda.get_device_properties() and
        # torch.cuda.current_device() in try/except RuntimeError → safe
        # defaults. FakeProps returns plausible H100-like values
        # (major=9, total_memory=80GB) so subsequent dtype selection logic
        # doesn't go down pathological branches.
        #
        # Namespacing: all injected names use _h3_* prefix to avoid collision
        # with any existing names in model_management.py.
        #
        # Idempotent: sentinel H3_TASK216c_torch_cuda_tolerance.
        # ------------------------------------------------------------------
        _SENTINEL_3 = "# H3_TASK216c_torch_cuda_tolerance"
        try:
            _mm_src = _mm_path.read_text()
            if _SENTINEL_3 in _mm_src:
                log.info("model_management.py: H3_TASK216c patch already applied — skipping")
            else:
                # Anchor: first `from __future__ import annotations` line.
                # Falls back to first `import psutil` if upstream changes
                # the __future__ import. Either anchor is BEFORE any
                # torch.cuda.* call in the file (lines 212+).
                _anchor_pattern = re.compile(
                    r"^(from __future__ import annotations|import psutil)\s*$",
                    re.MULTILINE,
                )
                _m3 = _anchor_pattern.search(_mm_src)
                if _m3:
                    _inject_pos = _m3.end()
                    # Inject a blank line + monkey-patch block. Indentation
                    # is intentionally 0 (top-level module code).
                    _injection = (
                        "\n"
                        "\n"
                        "# H3_TASK216c_torch_cuda_tolerance — DO NOT REMOVE\n"
                        "# Modal @modal.enter(snap=True) snapshots CPU+FS but NOT GPU state.\n"
                        "# After snap-restore, torch.cuda.* can raise RuntimeError(\"No CUDA GPUs ...\").\n"
                        "# Monkey-patch torch.cuda.get_device_properties and torch.cuda.current_device\n"
                        "# to return safe defaults on RuntimeError. Single point of tolerance —\n"
                        "# covers all current AND future call sites in model_management.py\n"
                        "# (Tasks #216d/#216e no longer needed).\n"
                        "try:\n"
                        "    import torch as _h3_torch_216c\n"
                        "    _h3_cuda_warned_216c = [False]\n"
                        "    def _h3_cuda_log_216c():\n"
                        "        if not _h3_cuda_warned_216c[0]:\n"
                        "            import logging as _logging\n"
                        "            _logging.warning(\n"
                        "                \"[H3_TASK216c] torch.cuda tolerance active \"\n"
                        "                \"(snap-restore GPU race — safe defaults returned)\"\n"
                        "            )\n"
                        "            _h3_cuda_warned_216c[0] = True\n"
                        "    _h3_orig_get_dev_props_216c = _h3_torch_216c.cuda.get_device_properties\n"
                        "    def _h3_safe_get_dev_props_216c(device):\n"
                        "        try:\n"
                        "            return _h3_orig_get_dev_props_216c(device)\n"
                        "        except RuntimeError:\n"
                        "            _h3_cuda_log_216c()\n"
                        "            class _FakeProps:\n"
                        "                major = 9\n"
                        "                minor = 0\n"
                        "                multi_processor_count = 132\n"
                        "                total_memory = 80 * 1024 * 1024 * 1024\n"
                        "            return _FakeProps()\n"
                        "    _h3_torch_216c.cuda.get_device_properties = _h3_safe_get_dev_props_216c\n"
                        "    _h3_orig_cur_dev_216c = _h3_torch_216c.cuda.current_device\n"
                        "    def _h3_safe_cur_dev_216c():\n"
                        "        try:\n"
                        "            return _h3_orig_cur_dev_216c()\n"
                        "        except RuntimeError:\n"
                        "            _h3_cuda_log_216c()\n"
                        "            return 0\n"
                        "    _h3_torch_216c.cuda.current_device = _h3_safe_cur_dev_216c\n"
                        "except Exception:\n"
                        "    pass\n"
                    )
                    _new_src = _mm_src[:_inject_pos] + _injection + _mm_src[_inject_pos:]
                    _mm_path.write_text(_new_src)
                    log.info("model_management.py: applied H3_TASK216c torch.cuda tolerance patch")
                else:
                    log.warning(
                        "model_management.py: H3_TASK216c anchor (from __future__ import annotations | "
                        "import psutil) not found — ComfyUI version may have changed; leaving file unchanged"
                    )
        except Exception as _patch_err3:
            log.warning(f"model_management.py: H3_TASK216c patcher failed: {_patch_err3}")

        # Launch ComfyUI on :8188 — same flags as prod serve() (R.128 baseline).
        import httpx as _httpx
        log_file = open("/tmp/comfy.log", "w")
        self._proc = subprocess.Popen(
            [python_bin, "/ComfyUI/main.py", "--listen", "127.0.0.1",
             "--port", "8188", "--disable-auto-launch", "--gpu-only",
             "--output-directory", "/modal-data/output"],
            stdout=log_file, stderr=subprocess.STDOUT,
        )
        log.info(f"ComfyUI PID: {self._proc.pid}")

        # Wait for ComfyUI ready (max 10 min for cold start)
        for i in range(600):
            try:
                r = _httpx.get("http://localhost:8188/system_stats", timeout=2)
                if r.status_code == 200:
                    log.info(f"ComfyUI ready after {i+1}s")
                    break
            except Exception:
                pass
            if i > 0 and i % 30 == 0:
                try:
                    with open("/tmp/comfy.log") as f:
                        log.info(f"[comfy.log tail @ {i}s]\n{f.read()[-2000:]}")
                except Exception:
                    pass
            time.sleep(1)
        else:
            try:
                with open("/tmp/comfy.log") as f:
                    log.error(f.read())
            except Exception:
                pass
            raise RuntimeError("ComfyUI startup timeout (H3Generator)")

        # Track initial output files for diff-after-execution
        self._initial_outputs = set(os.listdir("/modal-data/output"))
        log.info(f"H3Generator.setup() complete — initial_outputs={len(self._initial_outputs)}")

    @modal.method()
    def generate(self, workflow_json: str, image_b64: str, nonce: str) -> bytes:
        """R.119 + Task #218 — entry point invoked via .spawn() from api_run.

        Returns video bytes. Result is captured by Modal's FunctionCall
        infrastructure and made available via FunctionCall.from_id(call_id)
        in the api_result endpoint. Worker polls api_result with the
        call_id from the spawn response.

        Idempotency note: R.119 used modal.Dict for nonce-keyed atomic claim
        + result caching. Task #218 removes Dict entirely — concurrent submits
        with the same nonce are the WORKER's responsibility (worker keeps
        call_id, retries by re-polling same call_id rather than re-spawning;
        see Task #163 worker-side dedup). Modal FunctionCall itself is
        idempotent on (call_id) — multiple polls of the same call_id return
        the same result once available.
        """
        log.info(f"generate nonce={nonce}: starting (call_id-based)")
        t0 = time.time()
        result_bytes = self._run_workflow(workflow_json, image_b64, nonce)
        elapsed = time.time() - t0
        log.info(
            f"generate nonce={nonce}: DONE in {elapsed:.1f}s "
            f"({len(result_bytes)} bytes)"
        )
        return result_bytes

    @modal.fastapi_endpoint(method="POST")
    def api_run(self, payload: dict) -> dict:
        """Task #91 (2026-09-12) + Task #218 (2026-09-13): spawn-and-return.

        Spawns H3Generator.generate() via .spawn() — Modal returns IMMEDIATELY
        with a FunctionCall handle. Worker polls api_result with the returned
        call_id to retrieve the result.

        Why .spawn() instead of .local() (the previous Task #91 behavior):
          - .local() blocks the request thread until generate() returns —
            gates us on the 150s Modal gateway transport break. Cold-start
            can be >300s; any /api/run POST after that window hits a 303
            redirect with no result payload. Worker falls back to polling,
            which works but adds latency.
          - .spawn() returns the FunctionCall handle in <1s. The request
            thread is freed immediately. Generate runs to completion
            server-side, result captured by Modal's FunctionCall infra.
            Worker polls api_result with call_id — no 303 risk, no
            thread-blocking, no cold-start window coupling.

        URL: Modal auto-names this as <app>-<class>-<method>.modal.run:
            https://anton722451--comfyui-minimax-h3-h3generator-api-run.modal.run
        Worker calls this URL directly (no /api/run suffix — it's now the
        root of the endpoint).

        Signature uses Modal's recommended pattern — `payload: dict` directly
        (NOT `request: Request`). Modal/FastAPI introspects the signature and
        sees `dict` → treats the JSON body as the dict. With `Request` in the
        signature, FastAPI treats `request` as a query parameter (returns 422).
        sync `def` is fine here: payload parsing is automatic; .spawn() is
        non-blocking (returns immediately with FunctionCall handle).
        """
        from fastapi import HTTPException

        workflow_json = payload.get("workflow")
        # === FRAGILE-COMPENSATING-BUG (DO NOT FIX WITHOUT READING MEMORY) ===
        # Worker sends `imageData` (see apps/worker/src/processors/generation.processor.ts:336
        # `cachedRequestBody = JSON.stringify({..., imageData, ...}`), but THIS server reads
        # field name `image` below. Mismatch means image_b64 is ALWAYS empty for the current
        # R.119 worker path → fallback path at lines 912-924 (`if image_b64 and "easy
        # loadImageBase64" in str(workflow_json):`) does NOT fire.
        #
        # This is GOOD by accident: worker merge() (workflow-builder.ts:262
        # `matched.inputs[spec.input] = resolved`) already injects correct base64 into
        # node 220.base64_data BEFORE POST. Fallback would just re-set the same value.
        # But if fallback DID fire with a DIFFERENT base64 (e.g., wrong field name later
        # fixed without changing merge() output), it would silently overwrite the correct
        # value with empty/wrong data and break ref-image generation.
        #
        # DO NOT "fix" this mismatch to use `imageData` (or to send `image` from worker)
        # without simultaneously verifying:
        #   1. merge() still produces correct base64_data, AND
        #   2. fallback path semantics still match (re-set to same value, not corrupt)
        # Otherwise prod breaks. See memory entry
        # [[reelant-fragile-compensating-bug-image-data-vs-image-2026-09-09]] (Task #149).
        image_b64 = payload.get("image", "") or ""
        # === END FRAGILE-COMPENSATING-BUG MARKER ===
        nonce = payload.get("nonce")
        if not nonce:
            raise HTTPException(status_code=400, detail="missing nonce")
        if not workflow_json:
            raise HTTPException(status_code=400, detail="missing workflow")

        log.info(f"api_run nonce={nonce} spawning workflow")

        # Task #218: .spawn() (NOT .local()) — Modal SDK descriptor protocol.
        # .spawn() returns immediately with a FunctionCall handle. The heavy
        # work happens server-side; we don't block this request thread. The
        # 150s gateway transport break CANNOT fire because we return in <1s.
        try:
            call = self.generate.spawn(workflow_json, image_b64, nonce)
        except Exception as e:
            log.exception(f"api_run nonce={nonce} spawn failed")
            raise HTTPException(status_code=500, detail=f"spawn_failed: {str(e)[:500]}")

        log.info(f"api_run nonce={nonce} spawned call_id={call.object_id}")
        return {
            "status": "queued",
            "nonce": nonce,
            "call_id": call.object_id,
        }

    @modal.fastapi_endpoint(method="GET")
    def api_result(self, call_id: str) -> dict:
        """Task #218 (2026-09-13): poll call result via FunctionCall.from_id.

        URL: Modal auto-names this as <app>-<class>-<method>.modal.run.
        Worker hits `${MODAL_RESULT_URL}/?call_id={call_id}` (query-param
        style because @modal.fastapi_endpoint does NOT support path args —
        see [[modal-fastapi-endpoint-no-path-arg-2026-09-10]]).

        Returns (FastAPI JSON):
          200 + {video_b64, elapsed_s, size_bytes, call_id, status: "done"}
                — generate() completed successfully
          202 + {status: "running"} — call exists but generate() not done yet
          404 + {error: "call_not_found"} — call_id unknown or expired (Modal
                FunctionCall state is bounded; default ttl ~hours)
          500 + {error, detail} — generate() raised an exception; surface
                the message so worker can classify (REMOTE_FAILED, etc.)

        Implements Option A per Modal docs (community-blessed pattern for
        async result retrieval):
          https://modal.com/docs/guide/webhook-timeouts (see .spawn() section)
          https://modal.com/docs/guide/trigger-deployed-functions
        """
        import base64 as _b64
        from fastapi.responses import JSONResponse

        t0 = time.time()
        try:
            function_call = modal.FunctionCall.from_id(call_id)
        except modal.exception.NotFoundError:
            log.warning(f"api_result call_id={call_id}: not_found")
            return JSONResponse(
                {"error": "call_not_found", "call_id": call_id},
                status_code=404,
            )
        except Exception as e:
            log.exception(f"api_result call_id={call_id}: from_id failed")
            return JSONResponse(
                {"error": "from_id_failed", "detail": str(e)[:500]},
                status_code=500,
            )

        try:
            # Non-blocking poll — raises TimeoutError immediately if not done.
            # Modal internal state — no Dict eventual-consistency window
            # (Task #218 / Option A).
            result_bytes = function_call.get(timeout=0)
        except TimeoutError:
            return JSONResponse(
                {"status": "running", "call_id": call_id},
                status_code=202,
            )
        except modal.exception.NotFoundError:
            log.warning(f"api_result call_id={call_id}: expired between from_id and get")
            return JSONResponse(
                {"error": "call_not_found", "call_id": call_id},
                status_code=404,
            )
        except Exception as e:
            # generate() raised — Modal wraps the exception inside FunctionCall.get().
            # Surface the message; worker classifies via WorkerRemoteError.
            err_str = str(e)
            log.exception(f"api_result call_id={call_id}: generation failed: {err_str[:200]}")
            return JSONResponse(
                {"error": "generation_failed", "detail": err_str[:500], "call_id": call_id},
                status_code=500,
            )

        # Success — encode bytes to base64 (FastAPI jsonable_encoder fails on bytes,
        # same H.264 0xc3 issue that bit /api/status pre-fix). Worker consumes the
        # JSON envelope identically to Task #91's bytes-in-handler path.
        elapsed = time.time() - t0
        return {
            "status": "done",
            "video_b64": _b64.b64encode(result_bytes).decode("ascii"),
            "elapsed_s": round(elapsed, 2),
            "size_bytes": len(result_bytes),
            "call_id": call_id,
        }

    def _run_workflow(self, workflow_json: str, image_b64: str, nonce: str) -> bytes:
        """Submit workflow to local ComfyUI, poll history, return video bytes.

        flow:
          1. POST /api/prompt with {prompt: workflow_json, client_id: nonce}
          2. Poll /api/history/<prompt_id> every 3s until status.completed=true
             (max wait = local function timeout - 60s buffer)
          3. Extract output filename from history response
          4. Read /modal-data/output/
          5. Explicit volume.commit() — REPLACES R.137 watcher
          6. Return file bytes
        """
        import httpx as _httpx

        # 1. POST /api/prompt
        client_id = f"r119-{nonce}"
        if image_b64 and "easy loadImageBase64" in str(workflow_json):
            # === FRAGILE-COMPENSATING-BUG (see marker at line 583) ===
            # Worker normally injects base64_data into workflow BEFORE sending
            # via merge() (workflow-builder.ts:262). This branch is for callers
            # who provide image_b64 separately. It re-injects the same field —
            # normally idempotent. But if the upstream image_b64 is wrong/empty
            # AND the worker merge() was correct, this OVERWRITES with wrong value.
            # Currently safe because field-name mismatch (worker: imageData,
            # server: image) means image_b64 is always empty here. DO NOT enable
            # this path without re-verifying the data flow.
            # === END FRAGILE-COMPENSATING-BUG MARKER ===
            # The worker normally injects base64_data into the workflow before
            # sending. This branch handles cases where image_b64 is provided
            # separately. We do a minimal injection into the JSON if the
            # node exists — keeps the API ergonomic.
            try:
                wj = workflow_json if isinstance(workflow_json, dict) else _json.loads(workflow_json)
                for node_id, node in wj.items():
                    if isinstance(node, dict) and node.get("class_type") == "easy loadImageBase64":
                        node.setdefault("inputs", {})["base64_data"] = image_b64
                workflow_json = wj
            except Exception as e:
                log.warning(f"_run_workflow: image injection skipped — {e}")

        body = {"prompt": workflow_json, "client_id": client_id}
        r = _httpx.post(
            "http://localhost:8188/api/prompt",
            json=body,
            timeout=60,
        )
        if r.status_code != 200:
            raise RuntimeError(f"ComfyUI /api/prompt HTTP {r.status_code}: {r.text[:500]}")
        prompt_id = r.json().get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"No prompt_id in ComfyUI response: {r.text[:500]}")
        log.info(f"_run_workflow nonce={nonce} prompt_id={prompt_id}")

        # 2. Poll /api/history/<prompt_id>
        deadline = time.time() + 870  # 900s timeout - 30s buffer
        while time.time() < deadline:
            time.sleep(3)
            try:
                hr = _httpx.get(
                    f"http://localhost:8188/api/history/{prompt_id}",
                    timeout=10,
                )
                if hr.status_code != 200:
                    continue
                hist = hr.json().get(prompt_id)
                if not hist:
                    continue
                status = hist.get("status", {})
                if status.get("completed", False):
                    break
                if status.get("status_str") == "error":
                    raise RuntimeError(
                        f"ComfyUI workflow error: {hist.get('status', {})}"
                    )
            except _httpx.HTTPError:
                continue
        else:
            raise RuntimeError(f"ComfyUI poll timeout for prompt_id={prompt_id}")

        # 3. Find output file. ComfyUI stores outputs in hist["outputs"][node_id]["videos"|"images"].
        outputs = hist.get("outputs", {})
        out_filename = None
        out_subfolder = ""
        out_type = "output"
        for node_out in outputs.values():
            for kind in ("videos", "images", "gifs"):
                if kind in node_out and node_out[kind]:
                    first = node_out[kind][0]
                    out_filename = first.get("filename")
                    out_subfolder = first.get("subfolder", "")
                    out_type = first.get("type", "output")
                    break
            if out_filename:
                break

        if not out_filename:
            raise RuntimeError(
                f"No output found in ComfyUI history for prompt_id={prompt_id}: "
                f"{hist.get('outputs', {})}"
            )

        # 4. Read file. /ComfyUI/output is symlinked to /modal-data/output.
        out_path = os.path.join("/modal-data/output", out_subfolder, out_filename) \
            if out_subfolder else os.path.join("/modal-data/output", out_filename)
        if not os.path.exists(out_path):
            # Fall back: scan output dir for new files (handles edge cases where
            # ComfyUI writes to a different subfolder than reported).
            log.warning(
                f"_run_workflow: expected output at {out_path} but not found, "
                f"scanning /modal-data/output for new files"
            )
            current = set(os.listdir("/modal-data/output"))
            new = current - self._initial_outputs
            if not new:
                raise RuntimeError(
                    f"No output file at {out_path} and no new files in "
                    f"/modal-data/output since setup"
                )
            out_filename = sorted(new)[-1]  # most recent
            out_path = os.path.join("/modal-data/output", out_filename)

        # Wait briefly for file size to stabilize (ComfyUI may flush in chunks)
        for _ in range(10):
            sz1 = os.path.getsize(out_path)
            time.sleep(1)
            sz2 = os.path.getsize(out_path)
            if sz1 == sz2 and sz1 > 0:
                break

        with open(out_path, "rb") as f:
            video_bytes = f.read()
        log.info(f"_run_workflow nonce={nonce} read {len(video_bytes)} bytes from {out_path}")

        # 5. Explicit volume.commit() — replaces R.137 watcher. Sync from
        # caller's POV once commit() resolves; subsequent /modal-data/output
        # reads from other containers will see the file.
        try:
            h3_models_volume.commit()
        except Exception as e:
            # Non-fatal: bytes are already returned to caller; commit is for
            # Volume-level persistence (cross-container reads).
            log.warning(f"_run_workflow nonce={nonce} volume.commit() failed: {e}")

        return video_bytes

# =============================================================================
# DEAD CODE removed (Task #218 / Option A, 2026-09-13):
#   - `check_job_status(nonce)` — formerly R.119 cold-start-free status reader via
#     Modal Dict. Killed by Task #130 (2026-09-09) when /api/status inlined
#     jobs.get(); now doubly dead because Option A eliminates the Dict entirely
#     and replaces status with `FunctionCall.from_id(call_id).get(timeout=0)`.
#   - `jobs = modal.Dict.from_name(...)` removed at top of file — see header comment.
#
# If you ever need status without result payload, use:
#   call = FunctionCall.from_id(call_id)
#   try:
#       _ = call.get(timeout=0)   # 202 "running" or raises TimeoutError
#   except TimeoutError:
#       return {"status": "running", "call_id": call_id}
#
# =============================================================================
# Pre-warm: REMOVED 2026-08-31 (replaced by Modal-native autoscaler above).
# See B3 → B0 migration notes in MEMORY.
# =============================================================================
# Earlier design (B3): a scheduled function that pinged /system_stats every 4 min
# to keep the GPU container warm. This FAILED in production:
#   - Default Modal image lacks httpx (ModuleNotFoundError, container scaled down)
#   - Ping timeout (30-120s) < cold start (110s+) → scheduled ping races cold start
#   - Cron-style polling is fragile for keeping GPU state alive
#
# New design (B0): Modal-native autoscaler on serve() decorator:
#   - min_containers=0      → no idle cost, pay only for real requests
#   - scaledown_window=60   → die after 60s idle (serve() /api/run); /view uses 5s
#   - max_containers=20     → ceiling under burst
#   - buffer_containers=0   → REMOVED 2026-09-09 (was 1). See serve() comment
#                             + MEMORY [[modal-buffer-removed-permanently-2026-09-09]].
# (Modal SDK has no `scaleup_window`; autoscaler reacts to demand growth
# via its own internal heuristic — typically <1s.)
#
# Residual cold-start cost: first request after scaledown pays ~10s
# (memory snapshot restore, B1 2026-08-31). Mitigation: B1 worker
# IN_QUEUE_TIMEOUT_MS 10 → 25 min (worker safety net).
# =============================================================================


@app.local_entrypoint()
def main():
    log.info("=== Modal ComfyUI H3 app ===")
    log.info("Setup models:  modal run modal_comfyui_minimax_h3.py::setup_minimax_h3_models --hf-token hf_xxx")
    log.info("Setup Civitai LoRA: modal run modal_comfyui_minimax_h3.py::setup_civitai_lora_cli --model-id N --version-id N --file-id N --target-path models/loras/X.safetensors --sha256 ... --civitai-api-key KEY")
    log.info("Deploy:        modal deploy modal_comfyui_minimax_h3.py")
    log.info("Autoscaler:    min=0 max=20 buffer=1 scaledown=5s")


# =============================================================================
# Setup job: drop a Civitai LoRA into the same h3_models_volume
# =============================================================================
# Idempotent: if the destination already exists AND the SHA256 matches, exits OK.
# Re-running after the file is in place is a no-op (saves the 296 MB re-download).
#
# Why a separate function (vs adding to setup_minimax_h3_models):
#   - HuggingFace setup is large (~30 GB, 2-hour timeout). Adding/removing a
#     LoRA should NOT trigger re-download of the whole base model set.
#   - Idempotency contract differs: HF files are size-gated; LoRAs need
#     SHA256 verification (community uploads can drift).
# =============================================================================

CHUNK_SIZE_LORA = 8 * 1024 * 1024  # 8 MiB


def _sha256_file(path: str, chunk: int = 8 * 1024 * 1024) -> str:
    import hashlib as _hashlib

    h = _hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest().upper()


@app.function(
    image=image,
    volumes={"/modal-data": h3_models_volume},
    cpu=2,
    memory=4096,
    timeout=900,
    startup_timeout=600,
)
def setup_civitai_lora(
    model_id: int,
    version_id: int,
    file_id: int,
    target_path: str,
    sha256: str,
    civitai_api_key: str = "",
) -> dict:
    """Download a Civitai LoRA into the shared Modal Volume.

    Parameters
    ----------
    model_id, version_id, file_id
        Civitai identifiers — assembled from ``describe-lora`` manifest.
    target_path
        Path relative to the ComfyUI models root, e.g.
        ``"models/loras/MM-H3 - Blowjob v2.1.safetensors"``. The
        absolute destination is ``/modal-data/<target_path>``.
    sha256
        Expected SHA256 (uppercase hex, 64 chars). Verified after
        download; mismatch aborts and removes the partial file.
    civitai_api_key
        Bearer token. Required for NSFW or rate-limited public models.
        Read from env var CIVITAI_API_KEY if not passed.

    Returns
    -------
    dict with status, bytes written, sha256 (computed), path.
    """
    import urllib.request

    expected_sha = sha256.strip().upper()
    assert len(expected_sha) == 64, f"sha256 must be 64 hex chars, got {len(expected_sha)}"
    target_path = target_path.lstrip("/")
    dst = f"/modal-data/{target_path}"

    os.makedirs(os.path.dirname(dst), exist_ok=True)

    # Idempotency: file present + sha matches → done
    if os.path.exists(dst):
        h = _sha256_file(dst)
        if h == expected_sha:
            sz = os.path.getsize(dst)
            log.info(
                f"setup_civitai_lora: {target_path} already present "
                f"({sz / 1e6:.1f} MB, sha256 OK)"
            )
            return {
                "status": "ok_already_present",
                "path": target_path,
                "sizeBytes": sz,
                "sha256": h,
                "modelId": model_id,
                "versionId": version_id,
            }
        log.warning(
            f"setup_civitai_lora: {target_path} present but sha mismatch "
            f"(have {h[:12]}… want {expected_sha[:12]}…) — re-downloading"
        )
        os.unlink(dst)

    api_key = civitai_api_key or os.environ.get("CIVITAI_API_KEY", "")
    download_url = (
        f"https://civitai.com/api/download/models/{version_id}?fileId={file_id}"
    )
    # Cloudflare in front of Civitai returns a 403 HTML challenge for the
    # default `Python-urllib/x.y` User-Agent. Mimic curl to be allowed.
    headers = {
        "Accept": "application/octet-stream",
        "User-Agent": "curl/8.7.1",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    from urllib.parse import urlparse

    # Manual redirect loop — urllib's default HTTPRedirectHandler leaks
    # the original Authorization header to the redirected host (R2), which
    # then returns 400 "Missing x-amz-content-sha256" because its signed-URL
    # auth scheme rejects the leaked Bearer token.
    #
    # Solution: install an opener with a redirect handler that RE-RAISES
    # 30x as HTTPError with the Location header preserved, so urlopen
    # doesn't auto-follow. Our manual loop then follows with the right
    # headers (auth stripped on cross-host).
    class _NoFollowRedirect(urllib.request.HTTPRedirectHandler):
        def http_error_301(self, req, fp, code, msg, headers):
            return self._raise(req, fp, code, msg, headers)
        def http_error_302(self, req, fp, code, msg, headers):
            return self._raise(req, fp, code, msg, headers)
        def http_error_303(self, req, fp, code, msg, headers):
            return self._raise(req, fp, code, msg, headers)
        def http_error_307(self, req, fp, code, msg, headers):
            return self._raise(req, fp, code, msg, headers)
        def http_error_308(self, req, fp, code, msg, headers):
            return self._raise(req, fp, code, msg, headers)
        def _raise(self, req, fp, code, msg, headers):
            # Preserve headers on the HTTPError so caller can read Location
            err = urllib.error.HTTPError(
                req.full_url, code, msg, headers, fp
            )
            raise err

    opener = urllib.request.build_opener(_NoFollowRedirect())

    current_url = download_url
    current_headers = dict(headers)
    response = None
    try:
        for hop in range(5):  # max 5 hops, Civitai → R2 should be 1
            req = urllib.request.Request(current_url, headers=current_headers)
            try:
                response = opener.open(req, timeout=7200)
                # 2xx — done
                break
            except urllib.error.HTTPError as e:
                if e.code not in (301, 302, 303, 307, 308):
                    raise RuntimeError(
                        f"HTTP {e.code} from {current_url}: {e.reason}"
                    ) from e
                location = e.headers.get("Location")
                if not location:
                    raise RuntimeError(
                        f"HTTP {e.code} from {current_url} with no Location"
                    ) from e
                # Cross-host: strip Authorization (R2 uses its own sig).
                new_host = urlparse(location).netloc
                old_host = urlparse(current_url).netloc
                if new_host != old_host:
                    current_headers = {
                        k: v for k, v in current_headers.items()
                        if k != "Authorization"
                    }
                current_url = location
                log.info(
                    f"setup_civitai_lora: redirect hop {hop} → "
                    f"{urlparse(location).netloc}"
                )
                continue

        if response is None or response.status != 200:
            raise RuntimeError(f"unexpected final status from {current_url}")
        log.info(f"setup_civitai_lora: downloading {current_url}")

        tmp_path = f"{dst}.part"
        total = int(response.headers.get("Content-Length", "0") or 0)
        written = 0
        with open(tmp_path, "wb") as f:
            while True:
                chunk = response.read(CHUNK_SIZE_LORA)
                if not chunk:
                    break
                f.write(chunk)
                written += len(chunk)
                if total and written % (32 * CHUNK_SIZE_LORA) < CHUNK_SIZE_LORA:
                    pct = 100 * written / total
                    log.info(
                        f"setup_civitai_lora: {target_path} "
                        f"{written / 1e6:.1f}/{total / 1e6:.1f} MB ({pct:.1f}%)"
                    )
        response.close()
    except Exception as e:
        if os.path.exists(f"{dst}.part"):
            os.unlink(f"{dst}.part")
        raise RuntimeError(f"download failed: {e}") from e

    computed = _sha256_file(tmp_path)
    if computed != expected_sha:
        os.unlink(tmp_path)
        raise RuntimeError(
            f"sha256 mismatch: computed {computed[:12]}…, expected {expected_sha[:12]}…"
        )

    os.replace(tmp_path, dst)
    sz = os.path.getsize(dst)
    h3_models_volume.commit()
    log.info(
        f"setup_civitai_lora: committed {target_path} "
        f"({sz / 1e6:.1f} MB, sha256 OK)"
    )

    return {
        "status": "ok_downloaded",
        "path": target_path,
        "sizeBytes": sz,
        "sha256": computed,
        "modelId": model_id,
        "versionId": version_id,
    }


@app.local_entrypoint()
def setup_civitai_lora_cli(
    model_id: int,
    version_id: int,
    file_id: int,
    target_path: str,
    sha256: str,
    civitai_api_key: str = "",
) -> None:
    """CLI entrypoint. Invoked as::

        modal run modal_comfyui_minimax_h3.py::setup_civitai_lora_cli \\
            --model-id 2845331 --version-id 3235946 \\
            --file-id 3118341 \\
            --target-path "models/loras/MM-H3 - Blowjob v2.1.safetensors" \\
            --sha256 AEF6D0C6B758352FD4CFE302D3B9121FB0C18E470BDE4BDB2025229E1FEBEE6D \\
            --civitai-api-key "$CIVITAI_API_KEY"
    """
    import json as _json

    result = setup_civitai_lora.remote(
        model_id=model_id,
        version_id=version_id,
        file_id=file_id,
        target_path=target_path,
        sha256=sha256,
        civitai_api_key=civitai_api_key,
    )
    print(_json.dumps(result, indent=2))
