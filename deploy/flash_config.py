"""
flash_config.py — Production RunPod Serverless endpoint definition (phase 17).

Read-only module imported by:
  - deploy/deploy_prod.py (one-shot cutover script)
  - .github/workflows/deploy.yml (via `python -m deploy.flash_config`)
  - local REPL for introspection (`python -c "from deploy.flash_config import ep"`)

Pinned to the proven spike parameters (commit log: see /tmp/flash-spike/):
  - image:   antongoldobin/comfyui-ltx-video:latest
  - volume:  f3falnf3r0 (EU-RO-1, 200 GB)
  - gpu:     RTX 4090
  - region:  EU-RO-1
  - workers: (1, 3) warm pool
  - timeout: 30 min (matches Serverless max)

Endpoint name kept as "hon_cyan_tiger" to preserve the existing runsync URL
in Reelant-side ServerlessComfyUIProvider — no client changes needed.

To change image tag: set `ENDPOINT_IMAGE_TAG` env var (e.g. `:abc1234`).
"""

from __future__ import annotations

import os

from runpod_flash import DataCenter, Endpoint, GpuType, NetworkVolume  # noqa: F401  (DataCenter re-exported)

# -----------------------------------------------------------------------------
# Constants — change here, propagates everywhere
# -----------------------------------------------------------------------------

ENDPOINT_NAME = "hon_cyan_tiger"  # MUST match the existing prod endpoint

NETWORK_VOLUME_ID = "f3falnf3r0"  # EU-RO-1, 200 GB — DO NOT DELETE

DEFAULT_IMAGE = "antongoldobin/comfyui-ltx-video"
DEFAULT_IMAGE_TAG = "latest"

# (min, max) workers — `(1, 3)` keeps one warm worker to avoid cold starts
# while allowing bursty scaling. Bumping max increases GPU spend.
WORKERS_MIN = 1
WORKERS_MAX = 3

# execution_timeout_ms: serverless job hard limit. 30 min matches RunPod max
# for the LTX video pipeline (length 33 frames @ ~50 steps can take ~20 min).
EXECUTION_TIMEOUT_MS = 30 * 60 * 1000

# idle_timeout: serverless scales workers to zero after N seconds idle.
# 60 s keeps costs down — Reelant jobs come in bursts.
IDLE_TIMEOUT_SECONDS = 60

# GPU scarcity external risk: if RTX 4090 unavailable at EU-RO-1, Flash will
# queue. We don't multi-GPU-type the endpoint because that doubles deploy time
# and costs; if scarcity bites, switch GpuType manually.
GPU_TYPE = GpuType.NVIDIA_GEFORCE_RTX_4090
DATACENTER = DataCenter.EU_RO_1


# -----------------------------------------------------------------------------
# Derived Endpoint object (read-only at import time)
# -----------------------------------------------------------------------------

def _image() -> str:
    """Image reference. Override tag via env for SHA-pinned deploys."""
    tag = os.environ.get("ENDPOINT_IMAGE_TAG", DEFAULT_IMAGE_TAG)
    return f"{DEFAULT_IMAGE}:{tag}"


def build_endpoint() -> Endpoint:
    """Return a fresh Endpoint bound to the prod endpoint name."""
    return Endpoint(
        name=ENDPOINT_NAME,
        image=_image(),
        gpu=GPU_TYPE,
        workers=(WORKERS_MIN, WORKERS_MAX),
        idle_timeout=IDLE_TIMEOUT_SECONDS,
        volume=NetworkVolume(id=NETWORK_VOLUME_ID, dataCenterId=DATACENTER),
        datacenter=DATACENTER,
        execution_timeout_ms=EXECUTION_TIMEOUT_MS,
    )


# Module-level singleton — what scripts and CI import.
ep = build_endpoint()


# -----------------------------------------------------------------------------
# CLI introspection
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print(f"endpoint_name={ep.name}")
    print(f"image={ep.image}")
    print(f"gpu={ep.gpu}")
    print(f"datacenter={ep.datacenter}")
    print(f"workers=(min={ep.workers_min}, max={ep.workers_max})")
    print(f"idle_timeout={ep.idle_timeout}")
    print(f"execution_timeout_ms={ep.execution_timeout_ms}")
    print(f"volume={ep.volume}")
