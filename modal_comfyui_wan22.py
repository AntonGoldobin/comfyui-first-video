"""
modal_comfyui_wan22.py — Modal.com deployment of ComfyUI for Wan 2.1 + Phantom + Wan 2.2 I2V.

ACTIVE: 2026-09-27 — Style B (wan22-hearmeman-60fps) shipped alongside Style A.

A SEPARATE Modal app from comfyui-minimax-h3 so:
  - H3 endpoint stays untouched (no risk to existing production pipeline)
  - Wan models live on their own volume (don't share / fill H3's volume)
  - Independent scaling (Wan 14B has different VRAM/cold-start profile)

Mirrors the proven scaffold from modal_comfyui_minimax_h3.py:
  - sombi base image (sombi/comfyui:base-torch2.8.0-cu124)
  - /ComfyUI mount, --output-directory /modal-data/output
  - copy_dir_contents from Modal Volume to /ComfyUI/models/
  - 600s startup, 1800s timeout, 300s scaledown_window
  - @modal.concurrent(target_inputs=1, max_inputs=1) for serial GPU (BullMQ worker is
    concurrency=1, so Modal-side 1-way concurrent matches)
  - Rung 5 .spawn() + S3-only polling (same as H3)

Styles shipped:
  - wan22-face-portrait-nsfw  (Style A, ACTIVE 2026-09-26: Wan 2.1 + Phantom + NSFW, kijai stack)
  - wan22-hearmeman-60fps     (Style B, ACTIVE 2026-09-27: Wan 2.2 SFW, comfy-core native
                              WanImageToVideo + 2-pass KSamplerAdvanced + RIFE VFI 60 FPS,
                              fp8_scaled diffusion + lightx2v/Wan2.2-Distill-Loras)

Style deferred (gated on Style B E2E result):
  - wan22-hearmeman-60fps-nsfw (Style C, Wan 2.2 SFW + NSFW LoRA overlay)

Deploy:
  cd /Volumes/SSDNSKIY/VSCODE/comfyui-first-video
  modal deploy modal_comfyui_wan22.py
  modal run modal_comfyui_wan22.py::setup_wan22_models --hf-token hf_xxx
"""

import os
import time
import threading
import logging

import modal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("modal-comfyui-wan22")

app = modal.App("comfyui-wan22")
# SEPARATE volume — Wan 14B + T5-XXL ~22 GB for Style A only.
wan22_models_volume = modal.Volume.from_name(
    "comfyui-wan22-models", create_if_missing=True
)

# Same Rung 5 S3 secret as H3 — reuse reelant-s3 Modal Secret for result PUT.
_SECRETS = [modal.Secret.from_name("reelant-s3")]


# =============================================================================
# Image build — sombi base + ComfyUI v0.3.46 + custom nodes for Wan
# =============================================================================
image = (
    modal.Image.from_registry("sombi/comfyui:base-torch2.8.0-cu124")
    .run_commands(
        "apt-get update && apt-get install -y --no-install-recommends git wget ca-certificates python3 python3-venv python3-pip && rm -rf /var/lib/apt/lists/*",
        "python3 -m pip install --no-cache-dir --break-system-packages pip setuptools wheel || true",
    )
    .run_commands(
        "rm -rf /ComfyUI",
        "git clone --depth=1 --branch v0.33.2 https://github.com/comfyanonymous/ComfyUI /ComfyUI",
    )
    .run_commands(
        "pip install --no-cache-dir -r /ComfyUI/requirements.txt",
    )
    .run_commands(
        "git clone --depth=1 https://github.com/kijai/ComfyUI-WanVideoWrapper /ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper",
        "git clone --depth=1 https://github.com/kijai/ComfyUI-KJNodes /ComfyUI/custom_nodes/ComfyUI-KJNodes",
        "git clone --depth=1 https://github.com/rgthree/rgthree-comfy /ComfyUI/custom_nodes/rgthree-comfy",
        "git clone --depth=1 https://github.com/Fannovel16/ComfyUI-Frame-Interpolation /ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation",
        "git clone --depth=1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite /ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite",
        "git clone --depth=1 https://github.com/yolain/ComfyUI-Easy-Use /ComfyUI/custom_nodes/ComfyUI-Easy-Use",
        # Task #65 (2026-09-29): TeaCache — block-output caching, different mechanism
        # from SageAttention (no SDPA kernel swap, so no warm-container CUDA state
        # corruption). Expected 1.6-1.9x speedup vs fp8 baseline (131.7s → ~70-85s).
        #
        # Why welltop-cn instead of kijai WanVideoEasyCache: Style B workflow uses
        # comfy-core native WanImageToVideo + KSamplerAdvanced, NOT kijai WanVideoSampler.
        # kijai's WanVideoTeaCache/WanVideoEasyCache hook via cache_args into
        # WanVideoSampler only — they don't apply here.
        #
        # Known risks for this path:
        #   1. welltop-cn last commit 2025-07-12 (stale, but functional)
        #   2. Wan 2.2 tuning absent (Wan 2.1 coefficients only). Threshold scale
        #      is 10x different in published table — start at rel_l1_thresh=0.20
        #      for Wan 2.2 (conservative, identity first).
        #   3. Warm-container state may not reset between gens (similar to kijai
        #      Issue #371) — 3-trial E2E (1 cold + 2 warm) MUST verify quality.
        "git clone --depth=1 https://github.com/welltop-cn/ComfyUI-TeaCache /ComfyUI/custom_nodes/ComfyUI-TeaCache",
    )
    .run_commands(
        "for r in /ComfyUI/custom_nodes/*/requirements.txt; do "
        "[ -f \"$r\" ] && pip install --no-cache-dir -r \"$r\" || true; "
        "done",
        "pip install --no-cache-dir opencv-python imageio_ffmpeg",
        "pip install --no-cache-dir fastapi httpx 'starlette>=0.36' boto3",
        # Task #91 (2026-09-29): switch SageAttention from thu-ml main (2.2.0+) to
        # pip-released 1.0.6. Same kernel family, but prebuilt wheel (no JIT), no
        # build isolation, no TORCH_CUDA_ARCH_LIST. Same launch flag works
        # (--use-sage-attention). Pattern from customWF2026/modal_comfydeploy.
        # NOTE: SageAttention flag is currently DISABLED in launch (Task #85) due
        # to warm-container CUDA state corruption; kernel is installed for future
        # re-enable (single flag flip). Task #85b re-enable attempt FAILED at 3rd
        # warm gen (KSamplerAdvanced node 311).
        "pip install --no-cache-dir --break-system-packages sageattention==1.0.6",
        "pip install --no-cache-dir --break-system-packages transformers==4.56.0 huggingface_hub==0.36.2 torchaudio==2.8.0",
    )
    .entrypoint([])
)


# =============================================================================
# Setup job: download Wan 2.1 + Phantom + NSFW models to Modal Volume
# =============================================================================
WAN22_FILES = [
    # ─── Style A: Wan 2.1 + Phantom + NSFW (kijai stack) — shipped 2026-09-26 ───
    (
        "diffusion_models/Wan2_1-T2V-14B-Phantom_fp8_e4m3fn_scaled_KJ.safetensors",
        "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/T2V/Wan2_1-T2V-14B-Phantom_fp8_e4m3fn_scaled_KJ.safetensors",
        14_000_000_000,
    ),
    (
        "vae/wan_2.1_vae.safetensors",
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors",
        200_000_000,
    ),
    (
        "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        4_500_000_000,
    ),
    (
        "clip_vision/clip_vision_h.safetensors",
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors",
        1_000_000_000,
    ),
    (
        "loras/wan2.1_t2v_14b_lora_rank64_lightx2v_4step.safetensors",
        "https://huggingface.co/lightx2v/Wan2.1-Distill-Loras/resolve/main/wan2.1_t2v_14b_lora_rank64_lightx2v_4step.safetensors",
        500_000_000,
    ),
    (
        "loras/wan_cowgirl_v1.2.safetensors",
        "https://huggingface.co/mama2121/wan2.1lora/resolve/main/wan_cowgirl_v1.2.safetensors",
        250_000_000,
    ),
    # ─── Style B: Wan 2.2 I2V SFW (comfy-core native + RIFE 60 FPS) — 2026-09-27 ───
    # fp8_scaled (NOT _e4m3fn — _e4m3fn variant doesn't exist on HF for Wan 2.2 I2V).
    # Saves ~28 GB total vs fp16 (~29 GB → ~14 GB per file).
    (
        "diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
        "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
        14_000_000_000,
    ),
    (
        "diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
        "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
        14_000_000_000,
    ),
    # ─── Reference-clone A/B test (Task #60, 2026-09-28) ─────────────
    # fp16 variants of Wan 2.2 I2V diffusion models (same Wan 2.2 weights,
    # uncompressed). User's reference workflow (handover/reelant/2026-09-26_wan22_split/wan22-hearmeman-60fps.workflow.json,
    # 35 nodes) uses fp16; our current Style B uses fp8_scaled (smaller,
    # slightly lossy). New style 'wan22-hearmeman-60fps-ref' will switch
    # UNETLoader to these fp16 files via the workflow JSON. The copy loop
    # already picks up everything under /modal-data/models/diffusion_models/.
    (
        "diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors",
        "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors",
        28_000_000_000,
    ),
    (
        "diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors",
        "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors",
        28_000_000_000,
    ),
    # lightx2v 4-step distill LoRAs for Wan 2.2 I2V (high + low noise).
    # Repo: lightx2v/Wan2.2-Distill-Loras (NOT Wan2.2-I2V-A14B-Diffusers-distill-LoRA — 404).
    (
        "loras/wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors",
        "https://huggingface.co/lightx2v/Wan2.2-Distill-Loras/resolve/main/wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors",
        635_000_000,
    ),
    (
        "loras/wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors",
        "https://huggingface.co/lightx2v/Wan2.2-Distill-Loras/resolve/main/wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors",
        635_000_000,
    ),
    # ─── Style B (RIFE 60 FPS): RIFE VFI model, Task #15 (2026-09-27) ───
    # Fannovel16/ComfyUI-Frame-Interpolation vfi_utils.get_ckpt_container_path
    # returns `<custom_node>/ckpts/rife/` (NOT /ComfyUI/models/<sub>/). The
    # setup_wan22_models worker + the @modal.enter()'s copy loop stages the file
    # there explicitly below.
    (
        "rife/rife49.pth",
        "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/rife49.pth",
        21_000_000,
    ),
]


@app.function(
    image=image,
    volumes={"/modal-data": wan22_models_volume},
    cpu=4,
    memory=8192,
    timeout=7200,
    startup_timeout=600,
)
def setup_wan22_models(hf_token: str = "") -> dict:
    """Download Wan 2.1 + Phantom model set to Modal Volume. Idempotent. Total ~22 GB for Style A."""
    os.environ["HF_TOKEN"] = hf_token
    import urllib.request
    hdr = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    if os.path.isdir("/runpod-volume") and not os.path.islink("/runpod-volume"):
        os.system("rm -rf /runpod-volume && ln -s /modal-data /runpod-volume")

    log.info("=== Downloading Wan 2.1 + Phantom model set — Style A (~22 GB) ===")
    for rel_path, url, expected_min in WAN22_FILES:
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

    wan22_models_volume.commit()
    return {"status": "ok", "files": len(WAN22_FILES)}


# =============================================================================
# Task #176 (2026-09-18): Manual SigV4 PUT — bypass boto3 entirely.
# (Ported from H3 generator — same S3 endpoint, same SHA mismatch issue.)
# =============================================================================
import hashlib as _hashlib
import hmac as _hmac
import datetime as _datetime
import urllib.request as _urlreq
import urllib.error as _urlerr


def _sigv4_put(url, body, *, access_key, secret_key, region, bucket, key,
               content_type="video/mp4"):
    """PUT `body` to S3 via manual SigV4 (bypasses botocore entirely).

    Path-style addressing: URL is `https://host` (no path); we append
    `/{bucket}/{key}`. Returns (status_code, response_body_bytes).
    Raises urllib.error.HTTPError on 4xx/5xx (caller decides retry).
    """
    now = _datetime.datetime.utcnow()
    date_stamp = now.strftime("%Y%m%d")
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    payload_hash = _hashlib.sha256(body).hexdigest()

    # Parse host from URL.
    host = url.split("://", 1)[1].split("/", 1)[0]

    canonical_uri = f"/{bucket}/{key}"
    canonical_querystring = ""
    canonical_headers = (
        f"content-type:{content_type}\n"
        f"host:{host}\n"
        f"x-amz-content-sha256:{payload_hash}\n"
        f"x-amz-date:{amz_date}\n"
    )
    signed_headers = "content-type;host;x-amz-content-sha256;x-amz-date"

    canonical_request = (
        f"PUT\n"
        f"{canonical_uri}\n"
        f"{canonical_querystring}\n"
        f"{canonical_headers}\n"
        f"{signed_headers}\n"
        f"{payload_hash}"
    )

    credential_scope = f"{date_stamp}/{region}/s3/aws4_request"
    string_to_sign = (
        f"AWS4-HMAC-SHA256\n"
        f"{amz_date}\n"
        f"{credential_scope}\n"
        f"{_hashlib.sha256(canonical_request.encode()).hexdigest()}"
    )

    def _sign(key, msg):
        return _hmac.new(key, msg.encode(), _hashlib.sha256).digest()

    k_date = _sign(f"AWS4{secret_key}".encode(), date_stamp)
    k_region = _sign(k_date, region)
    k_service = _sign(k_region, "s3")
    k_signing = _sign(k_service, "aws4_request")
    signature = _hmac.new(k_signing, string_to_sign.encode(),
                          _hashlib.sha256).hexdigest()

    auth_header = (
        f"AWS4-HMAC-SHA256 Credential={access_key}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )

    req = _urlreq.Request(
        url + canonical_uri,
        data=body,
        method="PUT",
        headers={
            "Content-Type": content_type,
            "Content-Length": str(len(body)),
            "x-amz-content-sha256": payload_hash,
            "x-amz-date": amz_date,
            "Authorization": auth_header,
        },
    )
    try:
        with _urlreq.urlopen(req, timeout=120) as resp:
            return resp.status, resp.read()
    except _urlerr.HTTPError as e:
        return e.code, e.read()


def _write_failure_marker(nonce: str, *, error: str) -> None:
    """Task #178 (2026-09-18): fail-fast signal for Rung 5 .spawn() workers.

    Writes a small JSON sidecar at `generations/{nonce}.failed.json` so worker's
    pollS3ByGenId can HEAD it BEFORE polling `.mp4` and throw RemoteFailedError
    immediately (Layer B in Task #178).

    Reuses `_sigv4_put()` from Task #176.
    """
    import json as _json
    import time as _time
    key = f"generations/{nonce}.failed.json"
    body = _json.dumps({
        "nonce": nonce,
        "error": error[:500],
        "failed_at": _time.time(),
    }).encode("utf-8")
    try:
        status, _resp = _sigv4_put(
            url=os.environ["AWS_ENDPOINT_URL"],
            body=body,
            access_key=os.environ["AWS_ACCESS_KEY_ID"],
            secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            region=os.environ.get("AWS_REGION", "us-east-1"),
            bucket=os.environ["S3_BUCKET"],
            key=key,
            content_type="application/json",
        )
        log.info(f"failure marker PUT status={status} key={key}")
    except Exception as e:
        # Marker write MUST NOT mask the original RuntimeError — log loud, return.
        log.error(f"_write_failure_marker nonce={nonce} FAILED: {e!r}")


def _with_failure_marker(method):
    """Task #178: decorator that wraps `_run_workflow` (and similar) so any
    RuntimeError raised inside writes a fail-fast marker to S3 BEFORE re-raising.
    """
    import functools as _ft
    @_ft.wraps(method)
    def wrapper(self, *args, **kwargs):
        nonce = kwargs.get("nonce") or (args[2] if len(args) >= 3 else None)
        try:
            return method(self, *args, **kwargs)
        except RuntimeError as e:
            if nonce:
                _write_failure_marker(nonce, error=str(e))
            raise
    return wrapper


# =============================================================================
@app.cls(
    image=image,
    volumes={"/modal-data": wan22_models_volume},
    secrets=_SECRETS,
    cpu=4,
    memory=16384,
    timeout=1800,                  # class-level container lifetime cap
    enable_memory_snapshot=False,  # Task #176 (2026-09-18): snap=True caused snap-restore race that broke ComfyUI startup via _FakeProps.name AttributeError (and prior KeyError on memory_stats). WAN_TASK216 patches (a/b/c) remain as dead defense-in-depth. Cold-start ~22s→~44s, acceptable since min_containers=0 + scaledown_window=300 make cold-starts rare.
    min_containers=0,
    scaledown_window=300,  # Task H (2026-09-17): 60→300 — Task C worker pre-warm on /api/run guarantees container alive at submit time, so 60s scaledown_window (Task G) was overkill. Bump back to 300s for safety margin: covers burst pattern (Gen 1 → 7+ min polling → Gen 2) where worker is busy on Gen 1 result fetch when Gen 2 pre-warm fires. With Task C + max_containers=1 + scaledown_window=300: container cold-starts ONCE per idle gap (≥300s = 5 min), then reused across burst.
    max_containers=1,  # Task G (2026-09-17): 20→1 — community pattern for stateful GPU model workloads. Modal cold-start loop root cause was: parallel requests were being load-balanced to NEW container instances (Modal's burst autoscaler), each needing 30-60s cold-start. With max_containers=1, all burst requests go to the SINGLE instance, processed via @modal.concurrent(target_inputs=1, max_inputs=1) below. No horizontal scaling, no cold-start loops. Cold-start pays ONCE per idle gap, then reuses. Task H (2026-09-17): paired with Task C pre-warm — pre-warm at submit time means container is alive when POST lands, so 60s window no longer needed. Trade-off: if 5+ simultaneous gens, 5th waits in queue — acceptable since BullMQ already serializes 1-at-a-time. OOM follow-up (2026-09-20): 4→1 because worker is BullMQ-serial (concurrency:1), so Modal-side 4-way concurrent would have stacked overlapping Wan 22B UNet activations and OOMed at node 151 sampler on 3rd consecutive gen.
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
@modal.concurrent(target_inputs=1, max_inputs=1)  # OOM follow-up 2026-09-20: 4→1
class WanGenerator:
    @modal.enter(snap=False)  # Task #176: disable snap entirely — no snapshot, full cold-start every idle gap
    def setup(self):
        """Initialize GPU container once per cold start. Symlinks models,
        launches ComfyUI subprocess on :8188, waits for ready. Snapshotted."""
        import subprocess
        import shutil
        import pathlib
        import re

        log.info("=== WanGenerator.setup() — initializing GPU container ===")

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

        # ponytail: Task #56 (2026-09-28) — clip_vision was missing from the
        # copy loop, so comfy-core CLIPVisionLoader (#321) scanned an empty
        # /ComfyUI/models/clip_vision/ dir and rejected the workflow with
        # "clip_name: 'clip_vision_h.safetensors' not in []". File was already
        # on the Modal volume (models/clip_vision/) — just needed the copy.
        all_subs = (
            "diffusion_models", "text_encoders", "vae",
            "loras", "latent_upscale_models", "checkpoints",
            "clip_vision",
        )
        for sub in all_subs:
            os.makedirs(f"/modal-data/models/{sub}", exist_ok=True)
            os.makedirs(f"/ComfyUI/models/{sub}", exist_ok=True)

        log.info("Sync model copy (~50 GB / ~3 min)")
        # ponytail: cold-start RCA — verify whether copy loop re-runs inside setup()
        # despite @modal.enter(). Snapshots capture FS state from this point,
        # so the loop SHOULD be skipped on warm starts. If INSTR_COPY END appears on
        # warm starts, copy is being re-executed (bottleneck confirmed).
        copy_start_ts = time.monotonic()
        copy_start_wall = time.time()
        log.info(f"[INSTR_COPY] model copy START wall={copy_start_wall:.3f} monotonic={copy_start_ts:.3f}")
        for sub in all_subs:
            copy_dir_contents(f"/modal-data/models/{sub}", f"/ComfyUI/models/{sub}")
        # Task #15 (2026-09-27): Stage rife49.pth into ComfyUI-Frame-Interpolation
        # custom_node's own ckpts/rife/ folder. The custom_node reads from
        # `<custom_node>/ckpts/rife/rife49.pth` (Fannovel16/vfi_utils.py — relative
        # path to vfi_utils.py). NOT /ComfyUI/models/other/. Pre-bundle keeps RIFE
        # working at request-time without needing GitHub firewall egress.
        rife_ckpt_src = "/modal-data/models/rife/rife49.pth"
        rife_ckpt_dst_dir = "/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife"
        os.makedirs(rife_ckpt_dst_dir, exist_ok=True)
        rife_ckpt_dst = f"{rife_ckpt_dst_dir}/rife49.pth"
        if os.path.exists(rife_ckpt_src):
            if not os.path.exists(rife_ckpt_dst) or os.path.getsize(rife_ckpt_dst) != os.path.getsize(rife_ckpt_src):
                shutil.copy2(rife_ckpt_src, rife_ckpt_dst)
                log.info(f"[INSTR_RIFE] staged {rife_ckpt_src} -> {rife_ckpt_dst}")
            else:
                log.info(f"[INSTR_RIFE] already staged at {rife_ckpt_dst}")
        else:
            log.warning(f"[INSTR_RIFE] NOT FOUND: {rife_ckpt_src} — RIFE VFI may fall back to GitHub download at request-time")

        # Task #29 (2026-09-27): PATCH RIFE VFI custom_node line 238 — fix CPU/CUDA
        # tensor mismatch in torch.cat(output_frames). Root cause: RIFE accumulates
        # intermediate frames on CPU during inference (likely from clear_cache_after_n_frames
        # sync flush). Final torch.cat then errors with "tensors is on cpu, different from
        # other tensors on cuda:0". Fix: normalize device before cat. This is a 1-line
        # patch on the custom_node file baked into the Modal image.
        rife_init_py = "/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation/vfi_models/rife/__init__.py"
        if os.path.exists(rife_init_py):
            with open(rife_init_py, "r") as _f:
                _rife_src = _f.read()
            _marker = "# RIFE_PATCH_2026_09_27 device-normalize before cat"
            _dbg_marker = "# INSTR_RIFE_DEBUG: runtime probe"
            _patched_any = False
            if _marker not in _rife_src:
                # Insert before the torch.cat call. The original line is:
                #   out_tensor = torch.cat(output_frames, dim=0).to(torch.float32)
                _old = "out_tensor = torch.cat(output_frames, dim=0).to(torch.float32)"
                _new = (
                    f"{_marker}\n"
                    "        if len(output_frames) > 0:\n"
                    "            _device = output_frames[0].device\n"
                    "            output_frames = [f.to(_device) if f.device != _device else f for f in output_frames]\n"
                    "        out_tensor = torch.cat(output_frames, dim=0).to(torch.float32)"
                )
                if _old in _rife_src:
                    _rife_src = _rife_src.replace(_old, _new, 1)
                    log.info(f"[INSTR_RIFE_PATCH] applied device-normalize patch to {rife_init_py}")
                else:
                    log.warning(f"[INSTR_RIFE_PATCH] target line not found in {rife_init_py} — pattern may have changed upstream")

            if _dbg_marker not in _rife_src:
                # Task #35 (2026-09-27): inject a [INSTR_RIFE_DEBUG] print()
                # right after the first executable line inside vfi() so we can
                # see in Modal logs exactly what multiplier + frames.shape
                # reach RIFE at request time. Without this we cannot tell
                # whether multiplier=4 from DB arrives intact or is silently
                # coerced (e.g. coerced to int(1) by some upstream wrapper).
                _dbg_old = "        from .rife_arch import IFNet"
                _dbg_new = (
                    "        from .rife_arch import IFNet\n"
                    f"{_dbg_marker} — confirms multiplier reaches vfi() body.\n"
                    "        import sys as _dbg_sys, torch as _dbg_torch\n"
                    "        _dbg_sys.stdout.write(\n"
                    "            f\"[INSTR_RIFE_DEBUG] multiplier={multiplier!r} \"\n"
                    "            f\"type={type(multiplier).__name__} \"\n"
                    "            f\"ckpt_name={ckpt_name!r} \"\n"
                    "            f\"frames.shape={tuple(frames.shape) if hasattr(frames,'shape') else 'NA'} \"\n"
                    "            f\"frames.device={getattr(frames,'device','NA')} \"\n"
                    "            f\"dtype={dtype!r}\\n\"\n"
                    "        )\n"
                    "        _dbg_sys.stdout.flush()"
                )
                if _dbg_old in _rife_src:
                    _rife_src = _rife_src.replace(_dbg_old, _dbg_new, 1)
                    log.info(f"[INSTR_RIFE_DEBUG_PATCH] applied runtime-probe print to {rife_init_py}")
                else:
                    log.warning(f"[INSTR_RIFE_DEBUG_PATCH] target line not found in {rife_init_py} — pattern may have changed upstream")

            # Persist whatever we patched to disk (idempotent — both markers above
            # ensure re-runs are no-ops).
            with open(rife_init_py, "w") as _f:
                _f.write(_rife_src)
            log.info(f"[INSTR_RIFE] patch phase complete for {rife_init_py}")

            # Task #35 (2026-09-27): force module reload so ComfyUI sees the
            # patched bytes. Root cause: if `_run_workflow` ran BEFORE this
            # @enter() block completed (e.g. warm container reused), Python
            # already imported vfi_models.rife and cached `_model_cache` +
            # the `RIFE_VFI` class reference. Our file-on-disk patch is then
            # dead code on the next generate() call. `importlib.reload()` +
            # clearing the LRU cache invalidates the bound class. The 1-line
            # `try/except` keeps the warm path cheap if the module isn't yet
            # imported.
            try:
                import importlib
                import sys as _sys
                _sys.path.insert(0, "/ComfyUI/custom_nodes")
                import vfi_models.rife as _rife_mod
                importlib.reload(_rife_mod)
                # Re-publish on the parent package registry so ComfyUI's
                # NODE_CLASS_MAPPINGS lookup hits the freshly reloaded class.
                import vfi_models as _vm
                _vm.rife = _rife_mod
                log.info(f"[INSTR_RIFE_RELOAD] forced module reload of {rife_init_py}")
            except Exception as _reload_err:
                log.warning(f"[INSTR_RIFE_RELOAD] skipped (module not yet importable): {_reload_err}")
        else:
            log.warning(f"[INSTR_RIFE_PATCH] custom_node not found at {rife_init_py} — RIFE may not be installed in this image")

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
        # `torch._C._cuda_init()`. After a Modal @modal.enter()
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
        # `comfyui-wan22` crashlooping since v48 / 2026-09-13 10:24,
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
        _SENTINEL = "# WAN_TASK216_PATCH: deferred_cuda_init"
        try:
            _mm_src = _mm_path.read_text()
            if _SENTINEL in _mm_src:
                log.info("model_management.py: WAN_TASK216 patch already applied — skipping")
            else:
                _pattern = re.compile(
                    r"^(\s*)total_vram\s*=\s*get_total_memory\(get_torch_device\(\)\)\s*/\s*\(1024\s*\*\s*1024\)\s*$",
                    re.MULTILINE,
                )
                _m = _pattern.search(_mm_src)
                if _m:
                    _indent = _m.group(1)
                    _replacement = (
                        f"{_indent}# WAN_TASK216_PATCH: deferred_cuda_init\n"
                        f"{_indent}try:\n"
                        f"{_indent}    total_vram = get_total_memory(get_torch_device()) / (1024 * 1024)\n"
                        f"{_indent}except (RuntimeError, KeyError) as _e3_init_err:\n"
                        f"{_indent}    # Modal @modal.enter(snap=True) snapshots CPU+FS but NOT GPU state.\n"
                        f"{_indent}    # First import post-restore can race the GPU bind (RuntimeError — happens\n"
                        f"{_indent}    # when torch.cuda.current_device() raises 'No CUDA GPUs available').\n"
                        f"{_indent}    # KeyError raised when memory_stats() returns partial dict (snap-restore\n"
                        f"{_indent}    # state — reserved_bytes.all.current key may be missing). Defer both to 0;\n"
                        f"{_indent}    # total_vram is only read in Windows-only VRAM-reservation logic,\n"
                        f"{_indent}    # so on Linux this only affects the startup log line.\n"
                        f"{_indent}    # Task #216d (2026-09-14): broadened except tuple — get_total_memory\n"
                        f"{_indent}    # at line ~403 crashes with KeyError on stats['reserved_bytes.all.current']\n"
                        f"{_indent}    # when GPU telemetry dict is missing fields during cold-start snap-restore.\n"
                        f"{_indent}    # TypeError/AttributeError also covered defensively for adjacent dict-access\n"
                        f"{_indent}    # failures (None stats, stale device handle, etc.).\n"
                        f"{_indent}    total_vram = 0\n"
                        f"{_indent}    logging.warning(\n"
                        f"{_indent}        \"WAN_TASK216d: deferred init in comfy.model_management caught %s: %s\",\n"
                        f"{_indent}        type(_e3_init_err).__name__, _e3_init_err,\n"
                        f"{_indent}    )"
                    )
                    _new_src = _mm_src[:_m.start()] + _replacement + _mm_src[_m.end():]
                    _mm_path.write_text(_new_src)
                    log.info("model_management.py: applied WAN_TASK216 deferred-CUDA-init patch")
                else:
                    log.warning(
                        "model_management.py: WAN_TASK216 pattern not found — "
                        "ComfyUI version may have changed; leaving file unchanged"
                    )
        except Exception as _patch_err:
            log.warning(f"model_management.py: WAN_TASK216 patcher failed: {_patch_err}")

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
        _SENTINEL_2 = "# WAN_TASK216_PATCH_GET_TORCH_DEVICE: deferred_cuda_init"
        try:
            _mm_src = _mm_path.read_text()
            if _SENTINEL_2 in _mm_src:
                log.info("model_management.py: WAN_TASK216b patch already applied — skipping")
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
                        f"{_indent2}# WAN_TASK216_PATCH_GET_TORCH_DEVICE: deferred_cuda_init\n"
                        f"{_indent2}try:\n"
                        f"{_indent2}    return torch.device(torch.cuda.current_device())\n"
                        f"{_indent2}except RuntimeError as _e212_init_err:\n"
                        f"{_indent2}    # Modal @modal.enter() snapshots CPU+FS but NOT GPU state.\n"
                        f"{_indent2}    # get_torch_device() is called at module load by cuda_malloc_warning(),\n"
                        f"{_indent2}    # BEFORE the previously-patched line 363 ever runs. Fall back to CPU\n"
                        f"{_indent2}    # so ComfyUI import succeeds and the HTTP server can start; the next\n"
                        f"{_indent2}    # get_torch_device() call (after snap restore completes GPU bind) will\n"
                        f"{_indent2}    # return the real GPU device normally.\n"
                        f"{_indent2}    logging.warning(\n"
                        f"{_indent2}        \"WAN_TASK216b: deferred CUDA init in get_torch_device(): %s\",\n"
                        f"{_indent2}        _e212_init_err,\n"
                        f"{_indent2}    )\n"
                        f"{_indent2}    return torch.device(\"cpu\")"
                    )
                    _new_src = _mm_src[:_m2.start()] + _replacement2 + _mm_src[_m2.end():]
                    _mm_path.write_text(_new_src)
                    log.info("model_management.py: applied WAN_TASK216b deferred-CUDA-init patch (get_torch_device line 212)")
                else:
                    log.warning(
                        "model_management.py: WAN_TASK216b pattern not found — "
                        "ComfyUI version may have changed; leaving file unchanged"
                    )
        except Exception as _patch_err2:
            log.warning(f"model_management.py: WAN_TASK216b patcher failed: {_patch_err2}")

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
        # Idempotent: sentinel WAN_TASK216c_torch_cuda_tolerance.
        # ------------------------------------------------------------------
        _SENTINEL_3 = "# WAN_TASK216c_torch_cuda_tolerance"
        try:
            _mm_src = _mm_path.read_text()
            if _SENTINEL_3 in _mm_src:
                log.info("model_management.py: WAN_TASK216c patch already applied — skipping")
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
                        "# WAN_TASK216c_torch_cuda_tolerance — DO NOT REMOVE\n"
                        "# Modal @modal.enter() snapshots CPU+FS but NOT GPU state.\n"
                        "# After snap-restore, torch.cuda.* can raise RuntimeError(\"No CUDA GPUs ...\").\n"
                        "# Monkey-patch torch.cuda.get_device_properties and torch.cuda.current_device\n"
                        "# to return safe defaults on RuntimeError. Single point of tolerance —\n"
                        "# covers all current AND future call sites in model_management.py\n"
                        "# (Tasks #216d/#216e no longer needed).\n"
                        "try:\n"
                        "    import torch as _wan_torch_216c\n"
                        "    _wan_cuda_warned_216c = [False]\n"
                        "    def _wan_cuda_log_216c():\n"
                        "        if not _wan_cuda_warned_216c[0]:\n"
                        "            import logging as _logging\n"
                        "            _logging.warning(\n"
                        "                \"[WAN_TASK216c] torch.cuda tolerance active \"\n"
                        "                \"(snap-restore GPU race — safe defaults returned)\"\n"
                        "            )\n"
                        "            _wan_cuda_warned_216c[0] = True\n"
                        "    _wan_orig_get_dev_props_216c = _wan_torch_216c.cuda.get_device_properties\n"
                        "    def _wan_safe_get_dev_props_216c(device):\n"
                        "        try:\n"
                        "            return _wan_orig_get_dev_props_216c(device)\n"
                        "        except RuntimeError:\n"
                        "            _wan_cuda_log_216c()\n"
                        "            class _FakeProps:\n"
                        "                major = 9\n"
                        "                minor = 0\n"
                        "                multi_processor_count = 132\n"
                        "                total_memory = 80 * 1024 * 1024 * 1024\n"
                        "                # WAN_TASK216e — .name added so torch.cuda.get_device_name()\n"
                        "                # (which calls get_device_properties(device).name) stops\n"
                        "                # crashing on snap-restore cold-start at\n"
                        "                # comfy/model_management.py:685 cuda_malloc_warning().\n"
                        "                # Defensive siblings for likely-next-access sites.\n"
                        "                name = \"NVIDIA H100-SXM5-80GB\"\n"
                        "                is_integrated = 0\n"
                        "                is_multi_gpu_board = 0\n"
                        "                L2_cache_size = 50 * 1024 * 1024  # 50 MB\n"
                        "            return _FakeProps()\n"
                        "    _wan_torch_216c.cuda.get_device_properties = _wan_safe_get_dev_props_216c\n"
                        "    _wan_orig_cur_dev_216c = _wan_torch_216c.cuda.current_device\n"
                        "    def _wan_safe_cur_dev_216c():\n"
                        "        try:\n"
                        "            return _wan_orig_cur_dev_216c()\n"
                        "        except RuntimeError:\n"
                        "            _wan_cuda_log_216c()\n"
                        "            return 0\n"
                        "    _wan_torch_216c.cuda.current_device = _wan_safe_cur_dev_216c\n"
                        "except Exception:\n"
                        "    pass\n"
                    )
                    _new_src = _mm_src[:_inject_pos] + _injection + _mm_src[_inject_pos:]
                    _mm_path.write_text(_new_src)
                    log.info("model_management.py: applied WAN_TASK216c torch.cuda tolerance patch")
                else:
                    log.warning(
                        "model_management.py: WAN_TASK216c anchor (from __future__ import annotations | "
                        "import psutil) not found — ComfyUI version may have changed; leaving file unchanged"
                    )
        except Exception as _patch_err3:
            log.warning(f"model_management.py: WAN_TASK216c patcher failed: {_patch_err3}")

        # Launch ComfyUI on :8188 — same flags as prod serve() (R.128 baseline).
        # Task #85: --use-sage-attention flag removed. Was enabled in Task #63 (commit
        # 319393f) and verified on cold container (90.7s Modal internal, 1.45x speedup
        # vs 131.7s baseline) — but Task #84 E2E found 33% failure rate on warm
        # containers (KSamplerAdvanced node 311 failures, root-caused to Sage kernel
        # CUDA state corruption). Upstream issues open: thu-ml/SageAttention #392
        # (CUDA-graph replay) + ComfyUI #6125 (--use-sage-attention CUDA illegal
        # memory access). Sage kernel remains installed at image-build (line 110);
        # re-enabling is a one-line change once those close. Meanwhile we accept
        # the ~131.7s baseline for reliability. Plan successor: Task #65 (TeaCache).
        # Task #92 (2026-09-29): --cache-none added. Disables ComfyUI's node output
        # caching (VAE decode, CLIP, LoraLoader, etc.). Fixes warm-container GPU
        # OOM at KSamplerAdvanced node 311 (Task #91 trial 2 exposed it) — cached
        # tensors from prior gen collided with new KSampler allocation (~28GB
        # dual-UNet + cached nodes exceeded H100 80GB ceiling). Trade-off: gen 1
        # is slower because VAE/CLIP re-encode every call (instead of cached). But
        # the alternative is 33-100% warm-fail rate. Acceptable since most real
        # traffic is cold-after-idle (Modal scales down containers after ~5min).
        import httpx as _httpx
        log_file = open("/tmp/comfy.log", "w")
        self._proc = subprocess.Popen(
            [python_bin, "/ComfyUI/main.py", "--listen", "127.0.0.1",
             "--port", "8188", "--disable-auto-launch", "--gpu-only",
             "--cache-none",
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
                    # Diagnostic (Task #42): dump /object_info to verify KJNodes registered
                    try:
                        oi = _httpx.get("http://localhost:8188/object_info", timeout=10)
                        if oi.status_code == 200:
                            info = oi.json()
                            names = sorted(info.keys())
                            log.info(f"[diag] ComfyUI loaded {len(names)} node types")
                            kj = [n for n in names if "ImageResizeKJv2" in n or "WanVideo" in n or "VHS" in n]
                            log.info(f"[diag] KJ/Wan/VHS nodes: {kj}")
                            has_iresizekjv2 = "ImageResizeKJv2" in names
                            log.info(f"[diag] ImageResizeKJv2 registered: {has_iresizekjv2}")
                            if not has_iresizekjv2:
                                # Dump first 50 node names for debugging
                                log.info(f"[diag] first 50 node names: {names[:50]}")
                                # Show what's in custom_nodes
                                try:
                                    import os as _os
                                    cn_path = "/ComfyUI/custom_nodes"
                                    cn_dirs = _os.listdir(cn_path)
                                    log.info(f"[diag] /ComfyUI/custom_nodes contents: {cn_dirs}")
                                    for d in cn_dirs:
                                        full = _os.path.join(cn_path, d)
                                        if _os.path.isdir(full):
                                            files = _os.listdir(full)[:5]
                                            log.info(f"[diag] {d}/ first files: {files}")
                                except Exception as _e1:
                                    log.warning(f"[diag] custom_nodes ls failed: {_e1}")
                                # Tail comfy.log for import errors
                                try:
                                    with open("/tmp/comfy.log") as f:
                                        full = f.read()
                                        log.info(f"[diag] comfy.log total size: {len(full)} chars")
                                        # Show last 8000 chars — Traceback fully
                                        tail = full[-8000:]
                                        log.info(f"[diag] comfy.log tail (8000 chars):\n{tail}")
                                except Exception as _e2:
                                    log.warning(f"[diag] comfy.log tail failed: {_e2}")
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
            raise RuntimeError("ComfyUI startup timeout (WanGenerator)")

        # Track initial output files for diff-after-execution
        self._initial_outputs = set(os.listdir("/modal-data/output"))
        log.info(f"WanGenerator.setup() complete — initial_outputs={len(self._initial_outputs)}")

    @modal.method()
    def generate(self, workflow_json: str, image_b64: str, nonce: str) -> bytes:
        """R.119 + Task #218 — entry point invoked via .spawn() from api_run.

        Returns video bytes. Result is captured by Modal's FunctionCall
        infrastructure and made available via FunctionCall.from_id(call_id)
        in the api_result endpoint. Worker polls api_result with the
        call_id from the spawn response.

        OOM follow-up (2026-09-20): torch.cuda.empty_cache() + gc.collect()
        at entry. Without this, consecutive requests on the same container
        accumulate cached tensors — Wan 22B UNet + HMNSFW/Turbo LoRA + audio
        VAE ran OOM at node 151 SamplerCustomAdvanced for 3rd gen on the
        same container (800×800×90 after 2 prior gens). Modal doesn't
        auto-release between concurrent inputs.

        Idempotency note: R.119 used modal.Dict for nonce-keyed atomic claim
        + result caching. Task #218 removes Dict entirely — concurrent submits
        with the same nonce are the WORKER's responsibility (worker keeps
        call_id, retries by re-polling same call_id rather than re-spawning;
        see Task #163 worker-side dedup). Modal FunctionCall itself is
        idempotent on (call_id) — multiple polls of the same call_id return
        the same result once available.
        """
        log.info(f"generate nonce={nonce}: starting (call_id-based)")
        # OOM follow-up (2026-09-20): release cached CUDA memory + GC before
        # _run_workflow loads Wan 22B UNet + LoRAs. Without this, 3rd consecutive
        # gen on the same container OOMed at node 151 sampler.
        try:
            import torch
            import gc as _gc
            _gc.collect()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            log.info(f"generate nonce={nonce}: pre-flight CUDA cleanup done "
                     f"(allocated={torch.cuda.memory_allocated()/1e9:.1f}GB, "
                     f"reserved={torch.cuda.memory_reserved()/1e9:.1f}GB)")
        except Exception as _oom_cleanup_err:
            log.warning(f"generate nonce={nonce}: pre-flight CUDA cleanup "
                        f"best-effort failed: {_oom_cleanup_err!r}")
        t0 = time.time()
        # Method-scope boto3 import (matching FastAPI/httpx convention at L36+).
        # Module-scope import breaks `modal deploy` local introspection —
        # Modal CLI imports this file in the local venv where boto3 is not
        # installed (it's only in the Modal image per L128). Defer to method
        # scope so introspection passes; boto3 is available at runtime because
        # the Modal image installs it. MEMORY
        # [[reelant-modal-spawn-no-s3-put-2026-09-18]].
        import boto3
        result_bytes = self._run_workflow(workflow_json, image_b64, nonce)
        elapsed = time.time() - t0
        log.info(
            f"generate nonce={nonce}: DONE in {elapsed:.1f}s "
            f"({len(result_bytes)} bytes) -- uploading to S3"
        )

        # Rung 5 (2026-09-18): S3 PUT inside generate() so result is durable
        # BEFORE .spawn() caller's HTTP response returns. Worker polls S3 by
        # `generations/{nonce}.mp4` (see worker pollS3ByGenId). PUT is
        # idempotent on key — safe to retry.
        #
        # Task #176 (2026-09-18): boto3 PUT fails with XAmzContentSHA256Mismatch
        # from Modal's egress. Bypass boto3 entirely with manual SigV4 PUT via
        # stdlib urllib. See _sigv4_put() docstring for rationale.
        s3_key = f"generations/{nonce}.mp4"
        status, _resp = _sigv4_put(
            os.environ["AWS_ENDPOINT_URL"],
            result_bytes,
            access_key=os.environ["AWS_ACCESS_KEY_ID"],
            secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            region=os.environ.get("AWS_REGION", "us-east-1"),
            bucket=os.environ["S3_BUCKET"],
            key=s3_key,
            content_type="video/mp4",
        )
        if status not in (200, 204):
            raise RuntimeError(
                f"S3 PUT failed status={status} nonce={nonce} body={_resp!r}"
            )
        log.info(
            f"generate nonce={nonce} S3 PUT OK status={status} "
            f"s3://{os.environ['S3_BUCKET']}/{s3_key} ({len(result_bytes)} bytes)"
        )

        # OOM follow-up (2026-09-20): release cached CUDA memory + GC after
        # workflow + S3 PUT. Worker is BullMQ-serial (concurrency:1), so the
        # Modal container will be reused for the NEXT gen — and that next gen
        # gets a clean GPU instead of inheriting this gen's cached tensors.
        # Without this, 3rd consecutive gen on the same container OOMed at
        # node 151 SamplerCustomAdvanced (Wan 22B UNet activations).
        try:
            import torch
            import gc as _gc_post
            _gc_post.collect()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            log.info(
                f"generate nonce={nonce}: post-flight CUDA cleanup done "
                f"(allocated={torch.cuda.memory_allocated()/1e9:.1f}GB, "
                f"reserved={torch.cuda.memory_reserved()/1e9:.1f}GB)"
            )
        except Exception as _oom_post_err:
            log.warning(
                f"generate nonce={nonce}: post-flight CUDA cleanup "
                f"best-effort failed: {_oom_post_err!r}"
            )

        return result_bytes

    # Rung 3 (2026-09-17): Result-via-S3. Modal writes MP4 to S3 BEFORE
    # returning, so the HTTP response is tiny (~200 bytes) and Modal's 150s
    # gateway limit becomes irrelevant — worker polls S3 by gen_id, not Modal.
    # See memory [[modal-rung-3-s3-result-delivery-2026-09-17]] for full design.
    #
    # Replaces Task #229 / Path Z's inline bytes-in-handles approach. Why:
    #   - Task #229's .remote() returning inline bytes = ~5-10MB body.
    #     Cold-start + inference = 144-272s. On >150s Modal gates with 303
    #     transport break, worker has to follow undocumented Location URL.
    #   - Rung 3: response is {status, s3_key, ...} = ~200 bytes. Even on
    #     cold-start 272s, Modal 303 fires but worker doesn't need to follow
    #     it — it just polls S3 by the expected key. Modal is OUT of the
    #     polling path entirely.
    @modal.fastapi_endpoint(method="POST")
    def api_run(self, payload: dict) -> dict:
        """Rung 3 (2026-09-17): write MP4 to S3 + return status.

        Calls WanGenerator.generate() via .remote() (NOT .spawn()) — Modal
        BLOCKS this request thread until generate() returns. Then PUTs the
        result to S3 at `generations/{nonce}.mp4`. Worker polls S3 by gen_id
        to know when result is ready.

        Three response shapes:
          - 200 + {status: "done", s3_key, nonce, elapsed_s, size_bytes}
                  — generate() OK AND S3 PUT succeeded (worker downloads from S3)
          - 200 + {status: "failed", error} — generate() raised or S3 PUT failed
                  (worker throws RemoteFailedError → auto-refund)
          - 303 + Location: <redirect_url> — Modal 150s gateway break (cold-start
                  exceeded 150s). Worker treats as "submitted, in flight" — polls
                  S3 by `generations/{nonce}.mp4`. Modal is OUT of polling path.

        Why S3 (and the inline-bytes history):
          - Task #229 (Rung 1, 2026-09-17): inline `video_b64` JSON envelope.
            Worked for warm path (<150s), failed at Modal 303 on cold-start —
            worker had to follow undocumented `__modal_attempt_token` Location URL.
            7 prior Tasks (#218, #228, #255, #260, #261, #265c-#265f) patched
            different corners of the 303 + polling contract, none structurally
            eliminated it.
          - Rung 3 (this code): Modal writes result to S3 BEFORE returning. S3
            is the durable, idempotent result store — worker polls S3 by
            `generations/{nonce}.mp4`. Modal function can run for its full
            24h timeout; HTTP response is just ~200 bytes. No 303 follow
            needed (worker doesn't care if Modal timed out at gateway — S3
            file either appears or doesn't).

        URL: Modal auto-names this as <app>-<class>-<method>.modal.run:
            https://anton722451--comfyui-wan22-h3generator-api-run.modal.run
        Worker calls this URL directly.

        Signature uses Modal's recommended pattern — `payload: dict` directly.
        sync `def` is fine here: .remote() blocks until generate() returns.
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

        log.info(f"api_run nonce={nonce} starting .spawn() generate (Rung 5: S3-only polling)")

        # Rung 5 (2026-09-18): .spawn() returns FunctionCall ID immediately.
        # HTTP response is <100ms regardless of cold-start or inference time.
        # Worker polls S3 by `generations/{nonce}.mp4` (see worker
        # pollS3ByGenId). Modal is OUT of the polling path entirely.
        # Modal's 150s gateway break no longer matters — S3 file either
        # appears (PUT inside generate()) or doesn't.
        try:
            fut = self.generate.spawn(workflow_json, image_b64, nonce)
        except Exception as e:
            log.exception(f"api_run nonce={nonce} spawn failed")
            return {
                "status": "failed",
                "nonce": nonce,
                "error": f"spawn_failed: {str(e)[:500]}",
            }

        log.info(
            f"api_run nonce={nonce} .spawn() returned call_id={fut.object_id} "
            f"(HTTP response <100ms — worker polls S3 from here)"
        )

        return {
            "status": "accepted",
            "nonce": nonce,
            "call_id": fut.object_id,
        }

    @_with_failure_marker
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
                # _json (the function-local alias) is only defined inside
                # _write_failure_marker(). Import json here so the harness
                # actually parses the JSON instead of NameError-warn-skipping.
                import json as _json
                wj = workflow_json if isinstance(workflow_json, dict) else _json.loads(workflow_json)
                for node_id, node in wj.items():
                    if isinstance(node, dict) and node.get("class_type") == "easy loadImageBase64":
                        node.setdefault("inputs", {})["base64_data"] = image_b64
                workflow_json = wj
            except Exception as e:
                log.warning(f"_run_workflow: image injection skipped — {e}")

        # Task #42: decode data:image/...;base64,... OR raw base64 in LoadImage
        # nodes to a temp file under /ComfyUI/input/, replace with filename.
        # Required because v0.33.2 ComfyUI's LoadImage reads a filename from
        # /ComfyUI/input/, NOT a data URI or raw base64. Worker merge() passes
        # params.image (raw base64) directly into LoadImage's `image` field.
        try:
            import json as _json2
            import base64 as _b64
            import re as _re
            import os as _os2
            wj2 = workflow_json if isinstance(workflow_json, dict) else _json2.loads(workflow_json)
            _os2.makedirs("/ComfyUI/input", exist_ok=True)
            _img_idx = 0  # unique suffix per LoadImage in workflow
            for _nid, _node in wj2.items():
                if not (isinstance(_node, dict) and _node.get("class_type") == "LoadImage"):
                    continue
                _inp = _node.setdefault("inputs", {})
                _img = _inp.get("image", "")
                if not isinstance(_img, str) or not _img:
                    continue
                # Strip optional data URI prefix → raw base64
                _b64str = _img
                _m = _re.match(r"^data:image/(jpeg|jpg|png|webp);base64,(.+)$", _img, flags=_re.DOTALL)
                if _m:
                    _b64str = _m.group(2)
                # Heuristic: if value has a real filename extension (.png/.jpg/etc)
                # AND length is short enough to be a path, treat as filename.
                # Otherwise try to decode as base64 (which can contain '/').
                _looks_like_filename = bool(_re.match(r"^[\w./-]+\.(png|jpg|jpeg|webp)$", _b64str, flags=_re.IGNORECASE)) and "/" in _b64str.split("/")[0] is False and len(_b64str) < 256
                if _looks_like_filename:
                    log.info(f"_run_workflow: node {_nid} image looks like filename, leaving as-is: {_b64str[:60]}")
                    continue
                if len(_b64str) < 200:
                    log.info(f"_run_workflow: node {_nid} image too short for base64 ({len(_b64str)}), leaving as-is: {_b64str[:60]}")
                    continue
                try:
                    _raw = _b64.b64decode(_b64str, validate=True)
                except Exception as _decode_err:
                    log.warning(f"_run_workflow: node {_nid} image not valid base64 ({_decode_err}), leaving as-is")
                    continue
                _ext = "jpg"
                # Sniff magic bytes for png/webp
                if _raw[:8] == b"\x89PNG\r\n\x1a\n":
                    _ext = "png"
                elif _raw[:4] == b"RIFF" and _raw[8:12] == b"WEBP":
                    _ext = "webp"
                _img_idx += 1
                _fname = f"wan22_{nonce}_{_img_idx}.{_ext}"
                _fpath = f"/ComfyUI/input/{_fname}"
                with open(_fpath, "wb") as _f:
                    _f.write(_raw)
                _inp["image"] = _fname
                log.info(f"_run_workflow: node {_nid} decoded base64 → {_fpath} ({len(_raw)} bytes)")
            workflow_json = wj2
        except Exception as _loadimg_err:
            log.warning(f"_run_workflow: LoadImage base64 decode skipped — {_loadimg_err}")

        # Task #38 (2026-09-27): log the actual prompt body fields that reach
        # ComfyUI for nodes 303 (RIFE VFI) + 94 (VHS_VideoCombine). After two
        # generation cycles showing 81 frames at 16 FPS despite RIFE patched
        # and DB correct, RIFE's vfi() body never executed (zero [INSTR_RIFE_DEBUG]
        # logs in 88acf974 run). Need to confirm whether worker sends the
        # expected multiplier=4 + frame_rate=60, or whether ComfyUI rewrites
        # the body server-side. Single log line — no side-effects.
        try:
            _wf = workflow_json if isinstance(workflow_json, dict) else _json.loads(workflow_json)
            _n303 = (_wf.get("303") or {}).get("inputs", {}) or {}
            _n94 = (_wf.get("94") or {}).get("inputs", {}) or {}
            log.info(
                f"[INSTR_PROMPT_DEBUG] node303.multiplier={_n303.get('multiplier')!r} "
                f"node303.frames={_n303.get('frames')!r} "
                f"node303.dtype={_n303.get('dtype')!r} "
                f"node94.images={_n94.get('images')!r} "
                f"node94.frame_rate={_n94.get('frame_rate')!r} "
                f"workflow_node_count={len(_wf)}"
            )
        except Exception as _prompt_dbg_err:
            log.warning(f"[INSTR_PROMPT_DEBUG] failed: {_prompt_dbg_err}")

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
            wan22_models_volume.commit()
        except Exception as e:
            # Non-fatal: bytes are already returned to caller; commit is for
            # Volume-level persistence (cross-container reads).
            log.warning(f"_run_workflow nonce={nonce} volume.commit() failed: {e}")

        return video_bytes

# =============================================================================
