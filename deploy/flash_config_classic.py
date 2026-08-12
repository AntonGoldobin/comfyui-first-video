"""
flash_config_classic.py — Production RunPod Serverless endpoint definition (classic, non-Flash).

Read-only module imported by:
  - scripts/deploy_classic.py (one-shot cutover script)
  - .github/workflows/deploy.yml (via `python -m deploy.flash_config_classic`)
  - local REPL for introspection (`python -c "from deploy.flash_config_classic import cfg"`)

Why classic (not Flash)? The classic runpod SDK 1.11+ reads these env vars at
MODULE IMPORT time (worker_state.WORKER_ID, rp_job.JOB_GET_URL):

  - RUNPOD_POD_ID  (used as the WORKER_ID)
  - RUNPOD_AI_API_KEY  (sent as Authorization header to /job-take/)
  - RUNPOD_WEBHOOK_GET_JOB  (polled repeatedly; must contain literal "$ID")

Flash runtime injects RUNPOD_ENDPOINT_ID + RUNPOD_FLASH_API_KEY instead,
which leaves WORKER_ID = random UUID and JOB_GET_URL containing "$ID"
placeholder rather than the endpoint id. The handler.py workaround is
fragile (env must be set BEFORE `import runpod`). Classic runtime injects
all three classic env vars directly, no workaround needed.

Pinned to the proven spike parameters (commit log: see /tmp/flash-spike/):
  - image:   antongoldobin/comfyui-ltx-video:latest
  - volume:  f3falnf3r0 (EU-RO-1, 200 GB)
  - gpu:     RTX 4090 (ADA_24)
  - region:  EU-RO-1
  - workers: (0, 3) burst pool
  - scaler:  QUEUE_DELAY (default for classic serverless)
  - timeout: 30 min execution, 30 s idle

Endpoint name kept as "hon_cyan_tiger" so the existing Reelant-side
ServerlessComfyUIProvider URL pattern stays valid.

To change image tag: set `ENDPOINT_IMAGE_TAG` env var (e.g. `:abc1234`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Tuple

# -----------------------------------------------------------------------------
# Constants — change here, propagates everywhere
# -----------------------------------------------------------------------------

ENDPOINT_NAME = "hon_cyan_tiger"  # MUST match the existing prod endpoint

NETWORK_VOLUME_ID = "f3falnf3r0"  # EU-RO-1, 200 GB — DO NOT DELETE

DEFAULT_IMAGE = "antongoldobin/comfyui-ltx-video"
DEFAULT_IMAGE_TAG = "latest"

# (min, max) workers — (0, 3) scales from zero so idle hours don't bleed
# GPU-seconds on a warm worker we never use. Trade-off: first job after idle
# pays a 30-60 s cold start. Reelant jobs come in bursts, so burst scaling
# handles it well.
WORKERS_MIN = 0
WORKERS_MAX = 3

# execution_timeout_ms: classic serverless job hard limit. 30 min matches
# RunPod max for the LTX video pipeline (length 33 frames @ ~50 steps
# can take ~20 min).
EXECUTION_TIMEOUT_MS = 30 * 60 * 1000

# idle_timeout: classic serverless scales workers to zero after N seconds idle.
# 30 s is the sweet spot — long enough to absorb rapid back-to-back jobs,
# short enough that a forgotten endpoint doesn't bleed overnight.
IDLE_TIMEOUT_SECONDS = 30

# Disk on the worker container. Classic workers need a lot if models aren't
# pre-staged on the network volume; our image bakes models in /comfyui so
# 20 GB is enough (ComfyUI + cached deps ~ 12 GB, headroom for ~5 min jobs).
CONTAINER_DISK_GB = 20

# GPU: RTX 4090 classic API id is "ADA_24". Region: EU-RO-1.
GPU_ID = "ADA_24"
DATACENTER_ID = "EU-RO-1"

# scalerType: QUEUE_DELAY is the default for classic serverless (Flash
# uses different scaler types). Don't change unless you know what
# scalerValue = ?
SCALER_TYPE = "QUEUE_DELAY"
SCALER_VALUE = 4  # default value used by RunPod for QUEUE_DELAY


# -----------------------------------------------------------------------------
# Config dataclass — printable, hashable, easy to inspect
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ClassicEndpointConfig:
    name: str
    image: str
    gpu_id: str
    datacenter_id: str
    network_volume_id: str
    workers_min: int
    workers_max: int
    idle_timeout: int
    execution_timeout_ms: int
    container_disk_gb: int
    scaler_type: str
    scaler_value: int
    # Classic serverless needs a "template" wrapping the image. The schema
    # accepts the template as an INLINE object on saveEndpoint (separate
    # saveTemplate creates pod templates — incompatible). The env dict here
    # is passed through to the worker container.
    template_env: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def template_name(self) -> str:
        # Template name includes the resource id so concurrent deploys don't
        # collide. Runtime creates a serverless template from this name.
        return f"{self.name}__classic"

    def to_endpoint_input(self) -> dict:
        """Payload for saveEndpoint mutation (classic).

        Serverless templates are passed INLINE (not as separate templateId) —
        RunPod rejects pod templates with id from saveTemplate. The runtime
        mints a new serverless template id from fields below.

        Required template fields: name, imageName, containerDiskInGb,
        dockerArgs, env. dockerArgs is required but empty string is fine
        since the entrypoint is in the image (Dockerfile ENTRYPOINT).
        """
        return {
            "name": self.name,
            "gpuIds": self.gpu_id,
            "networkVolumeId": self.network_volume_id,
            "workersMin": self.workers_min,
            "workersMax": self.workers_max,
            "idleTimeout": self.idle_timeout,
            "executionTimeoutMs": self.execution_timeout_ms,
            "locations": self.datacenter_id,
            "scalerType": self.scaler_type,
            "scalerValue": self.scaler_value,
            "template": {
                "name": self.template_name,
                "imageName": self.image,
                "containerDiskInGb": self.container_disk_gb,
                "dockerArgs": "",
                "env": [{"key": k, "value": v} for k, v in self.template_env],
            },
        }


def _image() -> str:
    tag = os.environ.get("ENDPOINT_IMAGE_TAG", DEFAULT_IMAGE_TAG)
    return f"{DEFAULT_IMAGE}:{tag}"


def _read_s3_env() -> list[tuple[str, str]]:
    """S3 env injected into the worker container so handler.py can upload
    ComfyUI outputs (mp4/png) and return presigned URLs back to Reelant.

    Without these env vars, handler.py's _upload_outputs() returns
    {"type": "local", "data": "file://..."} — Reelant's runpod-reconciler
    only accepts URLs starting with http (extractVideoUrl), so the
    generation "succeeds" but no video URL is ever recorded.

    Source: Reelant infra/.env.local (Golden Antelope S3, EU).
    Values are baked at deploy time so we don't burn a CI secret for
    non-secret config.
    """
    pairs = [
        ("S3_BUCKET", "reelant"),
        ("S3_ACCESS_KEY_ID", "qPlYfemQcxQXZlfL"),
        ("S3_SECRET_ACCESS_KEY", "VSuLUN1Wli2hRctt771qyW9tWinxpJbW"),
        ("S3_ENDPOINT_URL", "https://s3-api.cr.golden-antelope.ru"),
        ("S3_REGION", "us-east-1"),
        ("S3_FORCE_PATH_STYLE", "true"),
    ]
    return [(k, v) for k, v in pairs if v]


def build_cfg() -> ClassicEndpointConfig:
    """Return a fresh ClassicEndpointConfig for the prod endpoint."""
    return ClassicEndpointConfig(
        name=ENDPOINT_NAME,
        image=_image(),
        gpu_id=GPU_ID,
        datacenter_id=DATACENTER_ID,
        network_volume_id=NETWORK_VOLUME_ID,
        workers_min=WORKERS_MIN,
        workers_max=WORKERS_MAX,
        idle_timeout=IDLE_TIMEOUT_SECONDS,
        execution_timeout_ms=EXECUTION_TIMEOUT_MS,
        container_disk_gb=CONTAINER_DISK_GB,
        scaler_type=SCALER_TYPE,
        scaler_value=SCALER_VALUE,
        # S3 env (Golden Antelope bucket) injected for handler.py output
        # upload. RunPod-provided RUNPOD_AI_API_KEY / RUNPOD_POD_ID come
        # from the runtime automatically — no need to add them here.
        template_env=_read_s3_env(),
    )


# Module-level singleton — what scripts and CI import.
cfg = build_cfg()


# -----------------------------------------------------------------------------
# CLI introspection
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print(f"name={cfg.name}")
    print(f"image={cfg.image}")
    print(f"gpu={cfg.gpu_id}")
    print(f"datacenter={cfg.datacenter_id}")
    print(f"workers=(min={cfg.workers_min}, max={cfg.workers_max})")
    print(f"idle_timeout={cfg.idle_timeout}")
    print(f"execution_timeout_ms={cfg.execution_timeout_ms}")
    print(f"network_volume={cfg.network_volume_id}")
    print(f"template_name={cfg.template_name}")
    print(f"endpoint_input={json.dumps(cfg.to_endpoint_input(), indent=2)}")
