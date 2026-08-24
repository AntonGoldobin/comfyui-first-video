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
  - same 600s startup, 1800s timeout, 300s scaledown_window

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
import logging
import modal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("modal-comfyui-h3")

app = modal.App("comfyui-minimax-h3")
# SEPARATE volume — H3 (~30 GB) does NOT share with LTX-2's comfyui-models volume.
# This keeps LTX-2 deployment unaffected even if H3 fills the volume.
h3_models_volume = modal.Volume.from_name(
    "comfyui-minimax-h3-models", create_if_missing=True
)

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
        "rm -rf /ComfyUI/custom_nodes/ComfyUI-Easy-Use /ComfyUI/custom_nodes/ComfyUI-KJNodes /ComfyUI/custom_nodes/ComfyUI-LTXVideo",
        "git clone --depth=1 https://github.com/yolain/ComfyUI-Easy-Use /ComfyUI/custom_nodes/ComfyUI-Easy-Use",
        "git clone --depth=1 https://github.com/kijai/ComfyUI-KJNodes /ComfyUI/custom_nodes/ComfyUI-KJNodes",
        "git clone --depth=1 https://github.com/Lightricks/ComfyUI-LTXVideo /ComfyUI/custom_nodes/ComfyUI-LTXVideo",
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
        # Falls back gracefully if not installed; we install it for the perf boost.
        "pip install --no-cache-dir --break-system-packages sageattention || true",
        # Pin transformers/torchaudio/huggingface_hub to CUDA 12-compatible versions
        # (sombi base ships v5.x which conflicts with comfy.ldm.minimax imports).
        "pip install --no-cache-dir --break-system-packages transformers==4.56.0 huggingface_hub==0.36.2 torchaudio==2.8.0",
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
# Inference: ASGI app proxying to local ComfyUI (cloned from modal_comfyui.py)
# =============================================================================
@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/modal-data": h3_models_volume},
    cpu=4,
    memory=16384,
    timeout=1800,
    scaledown_window=300,
    startup_timeout=600,
)
@modal.asgi_app()
def serve():
    """ASGI app exposing ComfyUI HTTP API on Modal for H3."""
    from fastapi import FastAPI, Request, HTTPException, Response
    import httpx

    log.info("=== Starting ComfyUI (H3) on Modal ===")
    import subprocess
    # Symlink so any internal /runpod-volume paths in custom nodes work
    if os.path.isdir("/runpod-volume") and not os.path.islink("/runpod-volume"):
        os.system("rm -rf /runpod-volume && ln -s /modal-data /runpod-volume")
    # Find python
    python_bin = None
    for cand in [
        "/venv/bin/python3",
        "/venv/bin/python",
        "/opt/venv/bin/python",
        "/usr/bin/python3",
        "/usr/local/bin/python3",
    ]:
        if os.path.exists(cand) and os.access(cand, os.X_OK):
            python_bin = cand
            break
    if not python_bin:
        import shutil
        python_bin = shutil.which("python3") or shutil.which("python")
    if not python_bin:
        raise RuntimeError("No python interpreter found in container")
    log.info(f"Using python: {python_bin}")

    # Copy models from Modal Volume to /ComfyUI/models/. ComfyUI's scanner does
    # not follow symlinks across mount boundaries, so we must copy.
    import shutil

    def copy_dir_contents(src_dir: str, dst_dir: str):
        if not os.path.isdir(src_dir):
            return
        os.makedirs(dst_dir, exist_ok=True)
        for fname in os.listdir(dst_dir):
            p = f"{dst_dir}/{fname}"
            if os.path.islink(p):
                try:
                    p_real = os.path.realpath(p)
                    if p_real.startswith("/modal-data"):
                        log.info(f"Removing stale symlink {p} → {p_real}")
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
            log.info(f"Copied ({os.path.getsize(dst)/1e9:.2f} GB)")

    # H3 files live under text_encoders/, diffusion_models/, vae/, loras/
    for sub in ("diffusion_models", "text_encoders", "vae", "loras"):
        os.makedirs(f"/modal-data/models/{sub}", exist_ok=True)
        copy_dir_contents(f"/modal-data/models/{sub}", f"/ComfyUI/models/{sub}")

    # Persist output to Modal Volume so it survives container scaledown
    os.makedirs("/modal-data/output", exist_ok=True)
    if os.path.islink("/ComfyUI/output") or os.path.isdir("/ComfyUI/output"):
        if os.path.islink("/ComfyUI/output"):
            os.unlink("/ComfyUI/output")
        else:
            shutil.rmtree("/ComfyUI/output")
    os.symlink("/modal-data/output", "/ComfyUI/output")
    log.info("Linked /ComfyUI/output → /modal-data/output")

    # Launch ComfyUI on :8188
    log_file = open("/tmp/comfy.log", "w")
    proc = subprocess.Popen(
        [python_bin, "/ComfyUI/main.py", "--listen", "127.0.0.1", "--port", "8188",
         "--disable-auto-launch", "--gpu-only",
         "--output-directory", "/modal-data/output"],
        stdout=log_file, stderr=subprocess.STDOUT,
    )
    log.info(f"ComfyUI PID: {proc.pid}")

    # Wait for ComfyUI ready (max 10 min for cold start)
    for i in range(600):
        try:
            r = httpx.get("http://localhost:8188/system_stats", timeout=2)
            if r.status_code == 200:
                log.info(f"ComfyUI ready after {i+1}s")
                break
        except Exception:
            pass
        if i > 0 and i % 30 == 0:
            try:
                with open("/tmp/comfy.log") as f:
                    tail = f.read()[-2000:]
                log.info(f"[comfy.log tail @ {i}s]\n{tail}")
            except Exception:
                pass
        time.sleep(1)
    else:
        log.error("ComfyUI did not start within 600s")
        try:
            with open("/tmp/comfy.log") as f:
                log.error(f.read())
        except Exception:
            pass
        raise RuntimeError("ComfyUI startup timeout")

    web_app = FastAPI(title="ComfyUI MiniMax-H3 on Modal")

    async def proxy(request: Request, path: str) -> Response:
        body = await request.body()
        url = f"http://localhost:8188/{path}"
        skip = {"host", "content-length", "connection", "accept-encoding"}
        headers = {k: v for k, v in request.headers.items() if k.lower() not in skip}
        try:
            async with httpx.AsyncClient(timeout=1800) as client:
                r = await client.request(
                    method=request.method, url=url, content=body,
                    headers=headers, params=request.query_params,
                )
            return Response(
                content=r.content, status_code=r.status_code,
                media_type=r.headers.get("content-type", "application/octet-stream"),
            )
        except Exception as e:
            log.exception("ComfyUI proxy error")
            raise HTTPException(status_code=502, detail=f"ComfyUI proxy error: {e}")

    @web_app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def handle(request: Request, path: str):
        return await proxy(request, path)

    return web_app


@app.local_entrypoint()
def main():
    log.info("=== Modal ComfyUI H3 app ===")
    log.info("Setup models:  modal run modal_comfyui_minimax_h3.py::setup_minimax_h3_models --hf-token hf_xxx")
    log.info("Setup Civitai LoRA: modal run modal_comfyui_minimax_h3.py::setup_civitai_lora_cli --model-id N --version-id N --file-id N --target-path models/loras/X.safetensors --sha256 ... --civitai-api-key KEY")
    log.info("Deploy:        modal deploy modal_comfyui_minimax_h3.py")


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
