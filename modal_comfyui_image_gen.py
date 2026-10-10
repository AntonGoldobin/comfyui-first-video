"""
modal_comfyui_image_gen.py — Modal.com deployment of ComfyUI for image generation.

R-IMAGE-2.3 (2026-10-10) — Stage 1 of the krea-face pipeline.

Serves a single ComfyUI graph: Krea 2 turbo i2i (denoise 0.4) → FaceDetailer
(denoise 0.35) → face-preserved first-frame PNG. The output is consumed by
the worker as `first_frame` for the Wan 2.2 14B I2V stage.

Why a SEPARATE Modal app from comfyui-wan22 (R-IMAGE-2.3 doc):
  - Krea 2 + FaceDetailer use a different model stack (~17 GB Krea + SAM +
    Ultralytics face_yolov9c.pt) than Wan 14B (~22 GB UNet + VAE + T5-XXL).
    Sharing a single volume would mix models, slow cold-starts, and risk
    volume over-fill.
  - Stage 1 has a 90s timeout ceiling (R-IMAGE-2.2 plan) — much shorter than
    Wan 22B's 150s+. Different timeout profile = different @app.cls shape.
  - The worker's `submitJob()` (legacy path) already POSTs `/api/prompt` and
    polls `/api/history/<prompt_id>` — so this app exposes ComfyUI's NATIVE
    endpoints on :8188. No new endpoint surface, no Rung 3 S3 result
    pattern (Stage 1 is small enough to fit in the 150s gateway limit).

The worker's `pollUntilDoneLegacy` then GETs `/api/view?filename=...&type=output`
to download the PNG. R-IMAGE-2.2's `runKreaFaceStep` already wraps that flow
with `downloadVideoWithRetry` (the helper is content-agnostic, just badly named).

Deploy:
  cd /Volumes/SSDNSKIY/VSCODE/comfyui-first-video
  modal deploy modal_comfyui_image_gen.py
  modal run modal_comfyui_image_gen.py::setup_image_gen_models

After deploy, set the worker env:
  MODAL_KREA_FACE_BASE_URL=https://anton722451--comfyui-image-gen-comfyuiimagegen.modal.run
  (replace with the actual URL printed by `modal deploy`).
"""

import os
import pathlib
import time
import logging
import urllib.request

import modal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("modal-comfyui-image-gen")

app = modal.App("comfyui-image-gen")

# SEPARATE volume — Krea 2 + SAM + Ultralytics face detector ~17 GB.
# Distinct from comfyui-wan22-models (Wan 14B) and comfyui-h3-models (H3 22B).
image_gen_models_volume = modal.Volume.from_name(
    "comfyui-image-gen-models", create_if_missing=True
)


# =============================================================================
# Image build — sombi base + ComfyUI v0.3.46 + custom nodes for Krea 2 + FaceDetailer
# =============================================================================
image = (
    modal.Image.from_registry("sombi/comfyui:base-torch2.8.0-cu124")
    .run_commands(
        "apt-get update && apt-get install -y --no-install-recommends git wget ca-certificates python3 python3-venv python3-pip && rm -rf /var/lib/apt/lists/*",
        "python3 -m pip install --no-cache-dir --break-system-packages pip setuptools wheel || true",
    )
    .run_commands(
        "rm -rf /ComfyUI",
        "git clone --depth=1 --branch v0.3.46 https://github.com/comfyanonymous/ComfyUI /ComfyUI",
    )
    .run_commands(
        "pip install --no-cache-dir -r /ComfyUI/requirements.txt",
    )
    .run_commands(
        # ComfyUI-Easy-Use — provides `easy loadImageBase64` node (R-IMAGE-2.4
        # paramMapping: worker injects base64 here for the user's reference photo).
        "git clone --depth=1 https://github.com/yolain/ComfyUI-Easy-Use /ComfyUI/custom_nodes/ComfyUI-Easy-Use",
        # ComfyUI-Impact-Pack — provides FaceDetailer, SAMLoader, UltralyticsDetectorProvider.
        # Impact-Pack is the canonical host for these nodes; the user's Pornmaster KR2
        # workflow depends on them. ~50 MB clone, no extra model downloads at install time.
        "git clone --depth=1 https://github.com/ltdrdata/ComfyUI-Impact-Pack /ComfyUI/custom_nodes/ComfyUI-Impact-Pack",
        # Impact-Subpack — required for UltralyticsDetectorProvider, ensures the detector
        # backend loads correctly (subpack ships the actual ultralytics bindings).
        "git clone --depth=1 https://github.com/ltdrdata/ComfyUI-Impact-Subpack /ComfyUI/custom_nodes/ComfyUI-Impact-Subpack",
        # ComfyUI-segment-anything — provides SAMLoader (Segment Anything Model loader).
        # FaceDetailer uses SAM for the bbox refinement pass.
        "git clone --depth=1 https://github.com/storyicon/comfyui_segment_anything /ComfyUI/custom_nodes/comfyui_segment_anything",
    )
    .run_commands(
        "for r in /ComfyUI/custom_nodes/*/requirements.txt; do "
        "[ -f \"$r\" ] && pip install --no-cache-dir -r \"$r\" || true; "
        "done",
        "pip install --no-cache-dir opencv-python imageio_ffmpeg ultralytics",
    )
    .entrypoint([])
)


# =============================================================================
# Setup job: download Krea 2 + SAM + bbox + VAE models to Modal Volume
# =============================================================================
# Source workflow: Pornmaster_KR2_i2i_V1_FaceDetailer_2026_07_18
# (35 nodes → API-format, ~22 executable). Models referenced by class_type:
#   - UNETLoader (i2i + FaceDetailer stages): Pornmaster_krea2_Turbo_int8_convrot_v2.safetensors
#   - CLIPLoader (krea2 type): qwen3vl_4b_fp8_scaled.safetensors
#   - VAELoader: qwen_image_vae.safetensors
#   - SAMLoader: sam_vit_b_01ec64.pth
#   - UltralyticsDetectorProvider: bbox/face_yolov9c.pt (YOLOv9c face detector)
#
# CivitAI gating: Pornmaster_krea2_Turbo_int8_convrot_v2.safetensors is a
# civitai.com download (gated, model_id=2787374) — needs the actual Civitai
# model_version_id (3140696 per the workflow's metadata.json) + CIVITAI_API_KEY
# secret. Until Anton manually downloads the file (or pastes the real download
# URL into the IMAGE_GEN_FILES entry below), the UNet entry is COMMENTED OUT —
# deploys without it will fail at run-time with "Pornmaster_krea2_Turbo_int8
# _convrot_v2.safetensors not found" on the comfyui-image-gen-models volume.
# Everything else below is HF (publicly downloadable).
#
# Krea-2 base model (HF: Comfy-Org/Krea-2) provides a fallback UNet
# (krea2_raw_int8_convrot.safetensors) — usable for smoke testing the
# graph topology, but NOT a drop-in replacement for the Pornmaster fine-tune.
# Once the real Civitai UNet is downloaded locally, paste its public Civitai
# download URL (https://civitai.com/api/download/models/3140696?fileId=3020859
# or the equivalent) into the uncommented entry below.
IMAGE_GEN_FILES = [
    # Krea 2 turbo UNet — the actual model. CivitAI download (gated).
    # COMMENTED OUT until Anton provides the real Civitai URL:
    # (
    #     "diffusion_models/Pornmaster_krea2_Turbo_int8_convrot_v2.safetensors",
    #     "https://civitai.com/api/download/models/3140696?fileId=3020859",  # TBD — gated, needs CIVITAI_API_KEY
    #     6_500_000_000,  # ~6.5 GB int8 quantized
    # ),
    # Krea-2 BASE model (HF, public) — smoke-test fallback for graph topology.
    # Smaller (~3.5GB) than the gated Pornmaster UNet. Used by R-IMAGE-2.5
    # smoke before the real Civitai UNet is wired in.
    (
        "diffusion_models/krea2_raw_int8_convrot.safetensors",
        "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/diffusion_models/krea2_raw_int8_convrot.safetensors",
        3_000_000_000,
    ),
    # Qwen3-VL 4B (CLIP text encoder for Krea 2). HF download.
    # Note: HF repo Comfy-Org/Krea-2 ships qwen3vl_4b_bf16.safetensors (bf16,
    # not fp8_scaled — fp8_scaled is a derived quantisation not yet in the
    # Krea-2 repo). bf16 works with the CLIPLoader's type='krea2' setting.
    (
        "text_encoders/qwen3vl_4b_bf16.safetensors",
        "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_bf16.safetensors",
        8_000_000_000,
    ),
    # Qwen Image VAE. HF download (Krea-2 repo).
    (
        "vae/qwen_image_vae.safetensors",
        "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors",
        200_000_000,
    ),
    # SAM ViT-B — small/fast SAM variant. ComfyUI-segment-anything default.
    (
        "sams/sam_vit_b_01ec64.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        370_000_000,
    ),
    # YOLOv9c face detector — Impact-Pack's UltralyticsDetectorProvider loads
    # from /ComfyUI/models/ultralytics/bbox/ by default.
    (
        "ultralytics/bbox/face_yolov9c.pt",
        "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov9c.pt",
        60_000_000,
    ),
]


@app.function(
    image=image,
    volumes={"/modal-data": image_gen_models_volume},
    cpu=4,
    memory=8192,
    timeout=7200,
    startup_timeout=600,
)
def setup_image_gen_models(hf_token: str = "") -> dict:
    """Download Krea 2 + SAM + bbox + VAE models to Modal Volume. Idempotent. ~17 GB total.

    The civitai.com URL for the Krea 2 UNet requires CIVITAI_API_KEY to be set
    in the modal run environment (via the `civitai` Modal Secret). Without it,
    that single file fails to download; HF files download fine.
    """
    os.environ["HF_TOKEN"] = hf_token
    civitai_key = os.environ.get("CIVITAI_API_KEY", "")
    import urllib.error
    from urllib.parse import urlparse

    if os.path.isdir("/runpod-volume") and not os.path.islink("/runpod-volume"):
        os.system("rm -rf /runpod-volume && ln -s /modal-data /runpod-volume")

    log.info("=== Downloading image-gen model set (~17 GB) ===")
    for rel_path, url, expected_min in IMAGE_GEN_FILES:
        dst = f"/modal-data/models/{rel_path}"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst) and os.path.getsize(dst) > expected_min * 0.95:
            log.info(f"{rel_path}: already present ({os.path.getsize(dst)/1e9:.2f} GB)")
            continue
        log.info(f"Downloading {rel_path} from {url}")
        is_civitai = "civitai.com" in urlparse(url).netloc
        if is_civitai:
            # CivitAI: needs Bearer auth + manual redirect loop. Auto-follow
            # leaks Authorization to R2 → 400. Same pattern as wan22 setup.
            if not civitai_key:
                raise RuntimeError(
                    f"{rel_path}: civitai.com URL but CIVITAI_API_KEY not set "
                    "(load via `civitai` Modal Secret or pass --secret civitai=KEY)"
                )

            class _NoFollow(urllib.request.HTTPRedirectHandler):
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
                    raise urllib.error.HTTPError(
                        req.full_url, code, msg, headers, fp
                    )

            current_url = url
            current_headers = {
                "Authorization": f"Bearer {civitai_key}",
                "Accept": "application/octet-stream",
                "User-Agent": "curl/8.7.1",
            }
            opener = urllib.request.build_opener(_NoFollow())
            for hop in range(5):
                req = urllib.request.Request(current_url, headers=current_headers)
                try:
                    response = opener.open(req, timeout=7200)
                except urllib.error.HTTPError as e:
                    if e.code in (301, 302, 303, 307, 308):
                        loc = e.headers.get("Location")
                        if not loc:
                            raise RuntimeError(f"civitai redirect without Location: {e}")
                        log.info(f"  civitai hop {hop}: {e.code} → {loc[:80]}")
                        # Strip Authorization when redirecting off civitai.com
                        if "civitai.com" not in urlparse(loc).netloc:
                            current_headers.pop("Authorization", None)
                        current_url = loc
                        continue
                    raise
                break
            else:
                raise RuntimeError("civitai redirect chain > 5 hops")

            with open(dst, "wb") as f:
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
        else:
            hdr = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
            req = urllib.request.Request(url, headers=hdr)
            with urllib.request.urlopen(req, timeout=7200) as r, open(dst, "wb") as f:
                while True:
                    chunk = r.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
        log.info(f"{rel_path}: done ({os.path.getsize(dst)/1e9:.2f} GB)")

    log.info("=== image-gen setup complete ===")
    return {"ok": True, "files": [r[0] for r in IMAGE_GEN_FILES]}


# =============================================================================
# GPU class — runs ComfyUI on :8188, exposes the worker's expected endpoints
# (/api/prompt, /api/history, /api/view, /api/object_info, /system_stats)
# =============================================================================
@app.cls(
    image=image,
    volumes={"/modal-data": image_gen_models_volume},
    cpu=4,
    memory=16384,
    timeout=1800,
    enable_memory_snapshot=False,
    min_containers=0,
    scaledown_window=300,
    max_containers=1,  # Krea 2 FaceDetailer is VRAM-tight; serialize like Wan22.
    buffer_containers=0,
    startup_timeout=600,
    region="us-east",
    gpu="H100",
)
class ImageGenGenerator:
    @modal.enter(snap=False)
    def setup(self):
        """Initialize GPU container once per cold start. Symlinks models,
        launches ComfyUI subprocess on :8188, waits for ready."""
        import subprocess
        import shutil
        import httpx as _httpx

        log.info("=== ImageGenGenerator.setup() — initializing GPU container ===")

        # Find python interpreter (sombi image has /venv/bin/python3).
        python_bin = None
        for cand in [
            "/venv/bin/python3", "/venv/bin/python",
            "/opt/venv/bin/python", "/usr/bin/python3",
            "/usr/local/bin/python3",
        ]:
            if os.path.exists(cand) and os.access(cand, cand):
                python_bin = cand
                break
        if not python_bin:
            python_bin = shutil.which("python3") or shutil.which("python")
        if not python_bin:
            raise RuntimeError("No python interpreter found in container")
        log.info(f"Using python: {python_bin}")

        # Symlink so any internal /runpod-volume paths in custom nodes work.
        if os.path.isdir("/runpod-volume") and not os.path.islink("/runpod-volume"):
            os.system("rm -rf /runpod-volume && ln -s /modal-data /runpod-volume")

        # Copy model subdirs from /modal-data/models/ → /ComfyUI/models/.
        # ComfyUI scans /ComfyUI/models/<subdir>/ for ckpts at boot. We must
        # copy because Modal Volumes mount as a single dir — ComfyUI won't
        # re-scan /modal-data/models on its own.
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
            "loras", "sams", "ultralytics",
        )
        for sub in all_subs:
            os.makedirs(f"/modal-data/models/{sub}", exist_ok=True)
            os.makedirs(f"/ComfyUI/models/{sub}", exist_ok=True)

        copy_start = time.monotonic()
        log.info("[INSTR_COPY] model copy START")
        for sub in all_subs:
            copy_dir_contents(f"/modal-data/models/{sub}", f"/ComfyUI/models/{sub}")
        log.info(f"[INSTR_COPY] model copy END duration={time.monotonic()-copy_start:.1f}s")

        # Symlink /ComfyUI/output → /modal-data/output for SaveImage node.
        os.makedirs("/modal-data/output", exist_ok=True)
        if os.path.islink("/ComfyUI/output") or os.path.isdir("/ComfyUI/output"):
            if os.path.islink("/ComfyUI/output"):
                os.unlink("/ComfyUI/output")
            else:
                shutil.rmtree("/ComfyUI/output")
        os.symlink("/modal-data/output", "/ComfyUI/output")
        log.info("Linked /ComfyUI/output → /modal-data/output")

        # Stream ComfyUI stdout/stderr to Modal container stdout (visible in
        # `modal app logs`). Same pattern as wan22 app (line 1216-1225).
        try:
            if os.path.exists("/tmp/comfy.log") or os.path.islink("/tmp/comfy.log"):
                os.unlink("/tmp/comfy.log")
        except FileNotFoundError:
            pass
        try:
            os.symlink("/proc/1/fd/1", "/tmp/comfy.log")
            log.info("[INSTR_COMFYLOG] /tmp/comfy.log → /proc/1/fd/1")
        except FileExistsError:
            pass

        log_file = open("/tmp/comfy.log", "w")
        self._proc = subprocess.Popen(
            [python_bin, "/ComfyUI/main.py", "--listen", "127.0.0.1",
             "--port", "8188", "--disable-auto-launch", "--gpu-only",
             "--cache-none",
             "--output-directory", "/modal-data/output"],
            stdout=log_file, stderr=subprocess.STDOUT,
        )
        log.info(f"ComfyUI PID: {self._proc.pid}")

        # Wait for ComfyUI ready (max 10 min for cold start).
        for i in range(600):
            try:
                r = _httpx.get("http://localhost:8188/system_stats", timeout=2)
                if r.status_code == 200:
                    log.info(f"ComfyUI ready after {i+1}s")
                    # Diagnostic: dump /object_info to verify custom nodes registered.
                    try:
                        oi = _httpx.get("http://localhost:8188/object_info", timeout=10)
                        if oi.status_code == 200:
                            names = sorted(oi.json().keys())
                            log.info(f"[diag] ComfyUI loaded {len(names)} node types")
                            for needle in ("easy loadImageBase64", "FaceDetailer",
                                           "SAMLoader", "UltralyticsDetectorProvider",
                                           "KSampler", "UNETLoader", "VAEEncode"):
                                present = needle in names
                                log.info(f"[diag] {needle}: {present}")
                    except Exception as _diag_err:
                        log.warning(f"[diag] /object_info probe failed: {_diag_err}")
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
            raise RuntimeError("ComfyUI startup timeout (ImageGenGenerator)")

        # Initial output snapshot (for orphan cleanup if needed).
        self._initial_outputs = set(os.listdir("/modal-data/output"))
        log.info(f"ImageGenGenerator.setup() complete — initial_outputs={len(self._initial_outputs)}")


# Expose ComfyUI on :8188 to the worker. Modal's @modal.web_server proxies
# HTTPS traffic to the container's localhost:8188 — the worker's
# `submitJobLegacy` then POSTs to /api/prompt as if ComfyUI were local.
# This is the simplest path: no new endpoint surface, no Rung 3 plumbing.
# The trade-off is the 150s Modal gateway limit, but Stage 1's 90s ceiling
# (R-IMAGE-2.2 plan) means a cold-start that exceeds 150s is already a bug
# that should fail-fast + refund, not a 303 transport break.
@app.asgi()
def fastapi_app():
    """ASGI wrapper that proxies /api/* and /system_stats to ComfyUI on :8188."""
    import httpx
    from fastapi import FastAPI, Request
    from fastapi.responses import Response, JSONResponse

    web_app = FastAPI()
    client = httpx.AsyncClient(base_url="http://localhost:8188", timeout=300.0)

    @web_app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def proxy(request: Request, path: str):
        try:
            body = await request.body()
            upstream = await client.request(
                method=request.method,
                url=f"/{path}",
                content=body,
                headers={k: v for k, v in request.headers.items()
                         if k.lower() not in ("host", "content-length")},
                params=request.query_params,
            )
            return Response(
                content=upstream.content,
                status_code=upstream.status_code,
                headers={k: v for k, v in upstream.headers.items()
                         if k.lower() not in ("content-encoding", "transfer-encoding",
                                              "content-length")},
            )
        except httpx.RequestError as e:
            return JSONResponse(
                status_code=502,
                content={"error": f"upstream ComfyUI unreachable: {e!r}"},
            )

    return web_app
