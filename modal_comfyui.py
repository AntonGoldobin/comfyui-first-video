"""
modal_comfyui.py — Modal.com deployment of ComfyUI for LTX-Video.

Builds a fresh Modal image from a known-good base (sombi's comfyui base) instead
of pulling the antongoldobin/comfyui-ltx-video:latest image (which Modal cannot
pull reliably — pulled 18.6 GB but contains no Python at expected paths).

Modal API:
- POST /api/prompt { prompt: workflow, client_id } → { prompt_id }
- GET  /api/history/{prompt_id} → status
- GET  /api/view?filename=... → binary video

Replaces RunPod serverless endpoint lli3msvjgoswc2 (deleted by user 2026-08-17).
"""

import os
import time
import logging
import modal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("modal-comfyui")

app = modal.App("comfyui-ltx-video")
models_volume = modal.Volume.from_name("comfyui-models", create_if_missing=True)

# Build a fresh image from a known-good base. We use the sombi base image
# which has torch + CUDA + ComfyUI pre-installed at /comfyui. If that pull
# fails, we can fall back to pytorch/pytorch official.
# Note: sombi is the same base the antongoldobin image was built on.
image = (
    modal.Image.from_registry("sombi/comfyui:base-torch2.8.0-cu124")
    .run_commands(
        # Install git + python3 (sombi base uses uv-managed Python, not /usr/bin/python3)
        "apt-get update && apt-get install -y --no-install-recommends git wget ca-certificates python3 python3-venv python3-pip && rm -rf /var/lib/apt/lists/*",
        "python3 -m pip install --no-cache-dir --break-system-packages pip setuptools wheel || true",
    )
    .run_commands(
        # Clone custom nodes — sombi base uses /ComfyUI (capital C), not /comfyui
        "rm -rf /ComfyUI/custom_nodes/ComfyUI-KJNodes /ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite /ComfyUI/custom_nodes/rgthree-comfy /ComfyUI/custom_nodes/ComfyUI-LTXVideo",
        "git clone --depth=1 https://github.com/kijai/ComfyUI-KJNodes /ComfyUI/custom_nodes/ComfyUI-KJNodes || true",
        "git clone --depth=1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite /ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite || true",
        "git clone --depth=1 https://github.com/rgthree/rgthree-comfy /ComfyUI/custom_nodes/rgthree-comfy || true",
        # LTX-Video node pack — master branch (has LTXVGemmaCLIPModelLoader added Aug 2026).
        # The transformers==4.56.0 pin below prevents the CUDA 13 mismatch (sombi base has CUDA 12.8).
        "git clone --depth=1 https://github.com/Lightricks/ComfyUI-LTXVideo /ComfyUI/custom_nodes/ComfyUI-LTXVideo",
    )
    .run_commands(
        # Install ComfyUI deps + custom node deps
        "cd /ComfyUI && "
        "for r in custom_nodes/*/requirements.txt; do "
        "[ -f \"$r\" ] && pip install --no-cache-dir -r \"$r\" || true; "
        "done",
        "pip install --no-cache-dir opencv-python imageio_ffmpeg",
        # ASGI proxy deps (serve() uses FastAPI + httpx)
        "pip install --no-cache-dir fastapi httpx 'starlette>=0.36'",
        # CRITICAL: sombi base ships with:
        #   - transformers 5.4.0 (compiled for CUDA 13, fails with libcudart.so.12)
        #   - torchaudio 2.11.0 (compiled for CUDA 13, fails with libcudart.so.12)
        # The base image has CUDA 12.8 (torch 2.8.0+cu128, libcudart.so.12 only).
        # torchaudio 2.11.0 is loaded by comfy.ldm.lightricks.vae.audio_vae which is imported
        # transitively via ComfyUI-LTXVideo's latents.py. Without this pin, the entire
        # ComfyUI-LTXVideo node pack fails to import and LTXVGemmaCLIPModelLoader is missing.
        # Downgrade both to versions compiled for CUDA 12. transformers 4.56.0 has Gemma3.
        # Must come AFTER custom_nodes requirements.txt or pip will re-upgrade.
        "pip install --no-cache-dir transformers==4.56.0 huggingface_hub==0.36.2 torchaudio==2.8.0",
    )
    .entrypoint([])  # disable base image entrypoint; we start ComfyUI ourselves
)


# =============================================================================
# Setup job: download models to Modal Volume
# =============================================================================
@app.function(
    image=image,
    volumes={"/modal-data": models_volume},
    cpu=4,
    memory=8192,
    timeout=7200,
    startup_timeout=600,
)
def setup_models(hf_token: str = "") -> dict:
    """Download LTX-2 models to Modal Volume.

    Idempotent: skips files already present.
    """
    os.environ["HF_TOKEN"] = hf_token
    import urllib.request
    hdr = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    # Symlink /runpod-volume → /modal-data so existing tooling works
    if os.path.isdir("/runpod-volume") and not os.path.islink("/runpod-volume"):
        os.system("rm -rf /runpod-volume && ln -s /modal-data /runpod-volume")
    # Always download the FULL LTX-2 dev model (transformer, ~43 GB).
    # LTX-2 needs BOTH a transformer checkpoint AND a Gemma text encoder (separate files).
    log.info("Downloading ltx-2-19b-dev.safetensors (transformer, ~43 GB)")
    target = "/modal-data/models/checkpoints/ltx-2-19b-dev.safetensors"
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if os.path.exists(target) and os.path.getsize(target) > 40_000_000_000:
        log.info(f"Already present ({os.path.getsize(target)/1e9:.2f} GB)")
    else:
        url = "https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-dev.safetensors"
        log.info(f"Downloading {url}")
        req = urllib.request.Request(url, headers=hdr)
        with urllib.request.urlopen(req, timeout=7200) as r, open(target, "wb") as f:
            while True:
                chunk = r.read(8 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
            log.info(f"Done ({os.path.getsize(target)/1e9:.2f} GB)")

    # Download Gemma text encoder shards (~47 GB across 11 files).
    # These go in /modal-data/models/text_encoders/. The ComfyUI LTXVGemmaCLIPModelLoader
    # expects model path + tokenizer path. We'll point the loader at the first shard;
    # ComfyUI will use the index.json to find the rest.
    text_enc_dir = "/modal-data/models/text_encoders/ltx-2-gemma"
    os.makedirs(text_enc_dir, exist_ok=True)
    gemma_files = [
        ("model-00001-of-00011.safetensors", 1_690_000_000),
        ("model-00002-of-00011.safetensors", 4_990_000_000),
        ("model-00003-of-00011.safetensors", 4_840_000_000),
        ("model-00004-of-00011.safetensors", 4_950_000_000),
        ("model-00005-of-00011.safetensors", 4_910_000_000),
        ("model-00006-of-00011.safetensors", 4_950_000_000),
        ("model-00007-of-00011.safetensors", 4_910_000_000),
        ("model-00008-of-00011.safetensors", 4_950_000_000),
        ("model-00009-of-00011.safetensors", 4_910_000_000),
        ("model-00010-of-00011.safetensors", 4_950_000_000),
        ("model-00011-of-00011.safetensors", 2_690_000_000),
    ]
    for fname, expected_size in gemma_files:
        dst = f"{text_enc_dir}/{fname}"
        if os.path.exists(dst) and os.path.getsize(dst) > expected_size * 0.95:
            log.info(f"Gemma {fname}: already present ({os.path.getsize(dst)/1e9:.2f} GB)")
            continue
        url = f"https://huggingface.co/Lightricks/LTX-2/resolve/main/text_encoder/{fname}"
        log.info(f"Downloading Gemma {fname}")
        req = urllib.request.Request(url, headers=hdr)
        with urllib.request.urlopen(req, timeout=7200) as r, open(dst, "wb") as f:
            while True:
                chunk = r.read(8 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        log.info(f"Gemma {fname}: done ({os.path.getsize(dst)/1e9:.2f} GB)")
    # Also download the index.json for the text encoder
    idx_path = f"{text_enc_dir}/model.safetensors.index.json"
    if not os.path.exists(idx_path):
        url = "https://huggingface.co/Lightricks/LTX-2/resolve/main/text_encoder/model.safetensors.index.json"
        log.info(f"Downloading text_encoder index")
        req = urllib.request.Request(url, headers=hdr)
        with urllib.request.urlopen(req, timeout=300) as r, open(idx_path, "wb") as f:
            f.write(r.read())
    # Download config.json + generation_config.json (small, required by LTXVGemmaCLIPModelLoader)
    for cfg_fname in ["config.json", "generation_config.json"]:
        cfg_path = f"{text_enc_dir}/{cfg_fname}"
        if os.path.exists(cfg_path):
            log.info(f"{cfg_fname}: already present")
            continue
        url = f"https://huggingface.co/Lightricks/LTX-2/resolve/main/text_encoder/{cfg_fname}"
        log.info(f"Downloading {cfg_fname}")
        req = urllib.request.Request(url, headers=hdr)
        with urllib.request.urlopen(req, timeout=300) as r, open(cfg_path, "wb") as f:
            f.write(r.read())
    # Download tokenizer files into text_encoders/ltx-2-gemma/ (Gemma3 needs them next to the model)
    tokenizer_files = [
        ("tokenizer.json", 33_000_000),
        ("tokenizer.model", 4_700_000),
        ("tokenizer_config.json", 1_200_000),
        ("special_tokens_map.json", 700),
        ("processor_config.json", 70),
        ("preprocessor_config.json", 600),
        ("added_tokens.json", 35),
        ("chat_template.jinja", 1600),
    ]
    for fname, expected_size in tokenizer_files:
        dst = f"{text_enc_dir}/{fname}"
        if os.path.exists(dst) and os.path.getsize(dst) > expected_size * 0.95:
            log.info(f"tokenizer/{fname}: already present")
            continue
        url = f"https://huggingface.co/Lightricks/LTX-2/resolve/main/tokenizer/{fname}"
        log.info(f"Downloading tokenizer/{fname}")
        req = urllib.request.Request(url, headers=hdr)
        with urllib.request.urlopen(req, timeout=300) as r, open(dst, "wb") as f:
            f.write(r.read())
        log.info(f"tokenizer/{fname}: done ({os.path.getsize(dst)/1e6:.2f} MB)")
    rc = 0
    models_volume.commit()
    return {"status": "ok", "return_code": rc}


# =============================================================================
# Inference: ASGI app proxying to local ComfyUI
# =============================================================================
@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/modal-data": models_volume},
    cpu=4,
    memory=16384,
    timeout=1800,
    scaledown_window=300,
    startup_timeout=600,
)
@modal.asgi_app()
def serve():
    """ASGI app exposing ComfyUI HTTP API on Modal."""
    from fastapi import FastAPI, Request, HTTPException, Response
    import httpx

    log.info("=== Starting ComfyUI on Modal ===")
    import subprocess
    # Symlink so any internal /runpod-volume paths in custom nodes work
    if os.path.isdir("/runpod-volume") and not os.path.islink("/runpod-volume"):
        os.system("rm -rf /runpod-volume && ln -s /modal-data /runpod-volume")
    # Find python (robust — sombi base uses /venv not /opt/venv)
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
        # Last resort: search PATH
        import shutil
        python_bin = shutil.which("python3") or shutil.which("python")
    if not python_bin:
        raise RuntimeError("No python interpreter found in container")
    log.info(f"Using python: {python_bin}")
    # Copy (not symlink) checkpoints AND text_encoders from Modal Volume to /ComfyUI/models/.
    # ComfyUI's scanner does not follow symlinks across mount boundaries, so we must
    # copy. ~1 min for 43 GB. Skipped if a real file already exists at dst.
    import shutil

    def copy_dir_contents(src_dir: str, dst_dir: str):
        """Copy files from src_dir to dst_dir, skipping files that already exist as real files."""
        if not os.path.isdir(src_dir):
            return
        os.makedirs(dst_dir, exist_ok=True)
        # Remove stale symlinks
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
        # Copy files
        for fname in os.listdir(src_dir):
            src = f"{src_dir}/{fname}"
            dst = f"{dst_dir}/{fname}"
            if os.path.isfile(dst) and not os.path.islink(dst):
                continue
            if os.path.islink(dst):
                continue
            if os.path.isdir(src):
                # Recurse into subdirs (e.g. text_encoders/ltx-2-gemma/)
                copy_dir_contents(src, dst)
                continue
            log.info(f"Copying {src} → {dst}")
            shutil.copy2(src, dst)
            log.info(f"Copied ({os.path.getsize(dst)/1e9:.2f} GB)")

    os.makedirs("/modal-data/models/checkpoints", exist_ok=True)
    copy_dir_contents("/modal-data/models/checkpoints", "/ComfyUI/models/checkpoints")
    copy_dir_contents("/modal-data/models/text_encoders", "/ComfyUI/models/text_encoders")
    # Persist output files to Modal Volume so they survive container scaledown.
    # Without this, the worker's download after poll-completion races the 5min
    # scaledown_window and gets 404 because /ComfyUI/output is ephemeral.
    os.makedirs("/modal-data/output", exist_ok=True)
    if os.path.islink("/ComfyUI/output") or os.path.isdir("/ComfyUI/output"):
        if os.path.islink("/ComfyUI/output"):
            os.unlink("/ComfyUI/output")
        else:
            shutil.rmtree("/ComfyUI/output")
    os.symlink("/modal-data/output", "/ComfyUI/output")
    log.info("Linked /ComfyUI/output → /modal-data/output (persists across scaledown)")
    # Launch ComfyUI on :8188 (background, with logs to /tmp/comfy.log).
    # --output-directory points SaveVideo writes to /modal-data/output via the
    # symlink above, so completed videos persist across container restarts.
    log_file = open("/tmp/comfy.log", "w")
    proc = subprocess.Popen(
        [python_bin, "/ComfyUI/main.py", "--listen", "127.0.0.1", "--port", "8188",
         "--disable-auto-launch", "--gpu-only",
         "--output-directory", "/modal-data/output"],
        stdout=log_file, stderr=subprocess.STDOUT,
    )
    log.info(f"ComfyUI PID: {proc.pid}")

    # Wait for ComfyUI to be ready (max 10 min for cold start)
    for i in range(600):
        try:
            r = httpx.get("http://localhost:8188/system_stats", timeout=2)
            if r.status_code == 200:
                log.info(f"ComfyUI ready after {i+1}s")
                break
        except Exception:
            pass
        # Every 30s, tail /tmp/comfy.log to modal logs
        if i > 0 and i % 30 == 0:
            try:
                with open("/tmp/comfy.log") as f:
                    tail = f.read()[-2000:]
                log.info(f"[comfy.log tail @ {i}s]\n{tail}")
            except Exception:
                pass
        time.sleep(1)
    else:
        log.error("ComfyUI did not start within 600s — check /tmp/comfy.log")
        try:
            with open("/tmp/comfy.log") as f:
                log.error(f.read())
        except Exception:
            pass
        raise RuntimeError("ComfyUI startup timeout")

    web_app = FastAPI(title="ComfyUI LTX-Video on Modal")

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
    log.info("=== Modal ComfyUI app ===")
    log.info("Setup models: modal run modal_comfyui.py::setup_models --hf-token hf_xxx")
    log.info("Deploy:      modal deploy modal_comfyui.py")