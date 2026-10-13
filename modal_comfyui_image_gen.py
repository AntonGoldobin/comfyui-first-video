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
  - The worker's `submitJob()` (legacy path) already POSTs `/api/prompt` and
    polls `/api/history/<prompt_id>` — so this app exposes ComfyUI's NATIVE
    endpoints on :8188. No new endpoint surface, no Rung 3 S3 result
    pattern (Stage 1 is small enough to fit in the 150s gateway limit).

# R-IMAGE-2.5-fix3 (2026-10-10): Final fix. The intermediate attempt
# (R-IMAGE-2.5-fix3 v1) used @app.cls + @modal.enter + @modal.web_server,
# but Modal still killed the container with "Runner has been initializing
# for too long: 600 seconds" — because the @modal.enter method was
# blocking on a 600s `_httpx.get` wait loop. The CORRECT pattern (per
# modal-examples/10_integrations/streamlit/serve_streamlit.py) is:
#   @app.function() + @modal.web_server(8188) + subprocess.Popen(...) + return.
# The `subprocess.Popen` is non-blocking. The function returns after
# Popen + the ComfyUI-ready check. Modal's @web_server then keeps the
# container alive as long as the port is open. scaledown_window=300s.
# Also: changed `--listen 127.0.0.1` to `--listen 0.0.0.0` so Modal's
# proxy can reach ComfyUI from outside the container (per Modal docs:
# "@web_server requires binding to external network interface").

The worker's `pollUntilDoneLegacy` then GETs `/api/view?filename=...&type=output`
to download the PNG. R-IMAGE-2.2's `runKreaFaceStep` already wraps that flow
with `downloadVideoWithRetry` (the helper is content-agnostic, just badly named).

Deploy:
  cd /Volumes/SSDNSKIY/VSCODE/comfyui-first-video
  modal deploy modal_comfyui_image_gen.py
  modal run modal_comfyui_image_gen.py::setup_image_gen_models

After deploy, set the worker env:
  MODAL_KREA_FACE_BASE_URL=https://anton722451--comfyui-image-gen-run.modal.run
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
        # R-IMAGE-2.5-fix4 (2026-10-10): v0.3.46 → v0.39.0. v0.30+ adds the
        # `type: "krea2"` CLIPLoader recipe that the krea-face workflow uses
        # (Qwen3-VL-4B via comfy.text_encoders.qwen_image21). Without this
        # upgrade, validation fails with:
        #   Value not in list: type: 'krea2' not in [stable_diffusion, ...]
        # v0.39.0 is the latest stable release (published 2026-10-05).
        "git clone --depth=1 --branch v0.39.0 https://github.com/comfyanonymous/ComfyUI /ComfyUI",
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
        # ComfyUI-Lora-Manager — provides `Lora Loader (LoraManager)` node
        # (R-IMAGE-2.7b / 2026-10-12 — the new krea-image-v2 style uses
        # LoraManager nodes in its workflow so users can attach LoRAs at
        # generation time. LoraManager is a no-op when loras=[] (model + clip
        # pass through unchanged) so the basic smoke-test path works without
        # any LoRA files. When the user attaches LoRAs, the lora files must
        # land in models/loras/ on the volume.)
        "git clone --depth=1 https://github.com/willmiao/ComfyUI-Lora-Manager /ComfyUI/custom_nodes/ComfyUI-Lora-Manager",
        # ComfyUI-Krea was REMOVED (R-IMAGE-2.5-fix4 final, 2026-10-10):
        # krea-ai/comfyui-krea doesn't exist publicly. Instead, upgrading
        # ComfyUI core from v0.3.46 to v0.39.0 brings native `type: "krea2"`
        # CLIPLoader support (registered as CLIPType.KREA2 = 32 in
        # comfy/sd.py, with Qwen3-VL-4B recipe in comfy/text_encoders/
        # qwen_image21). No external plugin needed.
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
    # Krea 2 Turbo UNet (CivitAI gated) used by krea-image-v2.
    # Source: model 2726029 "Krea 2 Turbo Official Comfy-Org Checkpoints",
    # version 3091481 (krea2_turbo_int8_convrot), fileId 2971089 —
    # 13.2 GB. Picked 2026-10-13 after diagnosing that the previous URL
    # (models/3140696?fileId=3020859) actually points to model 2787374,
    # a "Workflows"-type ZIP of 3 ComfyUI workflow JSONs (~83 KB total),
    # NOT a safetensors. The user's local filename
    # `Pornmaster_krea2_Turbo_int8_convrot_v2.safetensors` is a renamed
    # copy of one of these official Comfy-Org variants; we use the
    # Comfy-Org original since it's the same authors as the working
    # krea2_raw_int8_convrot (HF fallback) and UNETLoader-compatible.
    # See handover: comfyui-first-video r-image-2-7b-... for diagnosis.
    (
        "diffusion_models/Pornmaster_krea2_Turbo_int8_convrot_v2.safetensors",
        "https://civitai.com/api/download/models/2726029?fileId=2971089",
        12_500_000_000,  # ~13.2 GB int8 (CivitAI-reported sizeKB=13_176_452)
    ),
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
    #
    # R-IMAGE-2.7b (2026-10-12) — added fp8_scaled entry for krea-image-v2.
    # No URL (file isn't on a public HF repo). The user uploads manually via:
    #   modal volume put comfyui-image-gen-models \
    #     <local>/qwen3vl_4b_fp8_scaled.safetensors \
    #     models/text_encoders/
    # The setup function's `if os.path.getsize(dst) > expected_min * 0.95`
    # check at line 220-223 skips re-download if the file is already there
    # AND large enough. expected_min is set below so that a successful
    # manual upload is preserved across redeploys.
    (
        "text_encoders/qwen3vl_4b_bf16.safetensors",
        "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_bf16.safetensors",
        8_000_000_000,
    ),
    (
        "text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
        "",  # manual upload — see comment above; setup() will skip download when URL is empty
        3_500_000_000,  # ~3.5 GB fp8 quantised
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
    secrets=[modal.Secret.from_name("civitai-api-key")],
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
        # R-IMAGE-2.7b (2026-10-12) — manual-upload entries (e.g. fp8_scaled
        # CLIP, NSFW fine-tunes) have an empty URL. Skip the download attempt
        # and warn loudly — the operator MUST `modal volume put` the file
        # BEFORE running setup with a populated entry, otherwise the
        # ComfyUI container will fail to find the model at runtime.
        if not url:
            log.warning(
                f"{rel_path}: no download URL — manual upload required. "
                f"Run: modal volume put comfyui-image-gen-models <local_file> models/{rel_path.split('/', 1)[0]}/"
            )
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
# GPU class — runs ComfyUI on :8188 + exposes the port via @modal.web_server
# =============================================================================
# R-IMAGE-2.5-fix3 (2026-10-10): switched from @app.function + @modal.web_server
# (which blocks on subprocess.Popen — Modal kills the container after 600s
# with "Runner has been initializing for too long") to @app.cls + @modal.enter
# + @modal.web_server(8188). The @modal.enter returns immediately after
# launching the ComfyUI subprocess (non-blocking). @modal.web_server on the
# cls exposes :8188 to Modal's proxy. The cls lifecycle (not a function
# while-loop) keeps the container alive between requests.
#
# Pattern matches modal_comfyui_wan22.py:496-518 exactly. Same @modal.enter
# setup, same ComfyUI subprocess launch flow, same port 8188.
@app.function(
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
@modal.web_server(8188, startup_timeout=600)
def run():
    """Run ComfyUI on :8188 with Krea 2 + FaceDetailer models.

    R-IMAGE-2.5-fix3 (2026-10-10): Reverted from @app.cls+@modal.enter+@modal.web_server
    back to the canonical @app.function+@modal.web_server pattern from
    modal-examples/10_integrations/streamlit/serve_streamlit.py. The
    function does THREE things then returns:
      1. Symlink /runpod-volume to /modal-data.
      2. Copy model subdirs from /modal-data/models/ to /ComfyUI/models/.
      3. subprocess.Popen launch ComfyUI on 0.0.0.0:8188 (NON-BLOCKING).
    The Popen call is non-blocking — function returns immediately after.
    Modal's @modal.web_server then registers the port. The ComfyUI
    subprocess is the long-running process; Modal keeps the container
    alive while the web_server port is open.

    The previous (R-IMAGE-2.5-fix3 first attempt) used @app.cls+@modal.enter
    and added a `while True: time.sleep(60)` liveness check. The while-loop
    BLOCKED the function from returning, which made Modal think the
    function was still "initializing" — Modal killed the container after
    600s with "Runner has been initializing for too long". Removing the
    while-loop and just letting `subprocess.Popen` be the long-running
    process matches the streamlit example exactly. No liveness check —
    if ComfyUI dies, the next request will fail and the worker will
    surface the error. (Modal's `scaledown_window=300` keeps the
    container alive for 5 min after the last request, so brief gaps
    between gens don't trigger cold-starts.)
    """
    import subprocess
    import shutil
    import httpx as _httpx

    log.info("=== run() — initializing GPU container ===")

    # Find python interpreter (sombi image has /venv/bin/python3).
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
    # R-IMAGE-2.5-fix3: --listen 0.0.0.0 (NOT 127.0.0.1) so Modal's
    # @modal.web_server proxy can reach ComfyUI from outside the
    # container. 127.0.0.1 only binds to the loopback interface.
    proc = subprocess.Popen(
        [python_bin, "/ComfyUI/main.py", "--listen", "0.0.0.0",
         "--port", "8188", "--disable-auto-launch", "--gpu-only",
         "--cache-none",
         "--output-directory", "/modal-data/output"],
        stdout=log_file, stderr=subprocess.STDOUT,
    )
    log.info(f"ComfyUI PID: {proc.pid}")

    # Wait for ComfyUI ready (max 10 min for cold start). This BLOCKS
    # `run()` for up to 10 min, but Modal's @modal.web_server timeout
    # is 600s, so if ComfyUI isn't ready in 10 min Modal will kill the
    # container with the same "Runner has been initializing for too long"
    # error. The first call (when Diag completes) typically takes ~30-60s
    # for model copy + ComfyUI ready. If we exceed 600s here, it means
    # something is wrong (e.g. missing model file, custom node crash).
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
        raise RuntimeError("ComfyUI startup timeout (run)")

    # Initial output snapshot (for orphan cleanup if needed).
    log.info(
        f"run() ready — initial_outputs={len(os.listdir('/modal-data/output'))}"
    )
    # R-IMAGE-2.5-fix3: return here — NO while-True loop. The ComfyUI
    # subprocess is the long-running process. Modal's @web_server keeps
    # the container alive while the port is open (scaledown_window=300
    # = 5 min after last request). This matches the streamlit example
    # pattern. Removing the while-loop fixed the "Runner has been
    # initializing for too long" kill — that error was caused by the
    # function never returning.

