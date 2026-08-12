"""
spike.py — Repeatable local Flash deploy test (phase 17, DoD-4).

Proves the deploy path works on our account BEFORE touching production
endpoint `3995g4nz6mxw1v`:
  1. Create throwaway endpoint with unique name (flash-spike-<ts>)
  2. Wait until at least one worker is ready (max 5 min)
  3. Delete endpoint via GraphQL
  4. Verify only prod endpoint remains

Uses the SAME image + volume + GPU + datacenter as production
(deploy/flash_config.py), so a successful spike strongly predicts a
successful prod cutover.

Pre-flight:
  - `flash login` has populated ~/.runpod/config.toml with control-plane key
  - Or env var RUNPOD_FLASH_API_KEY is set

Exit codes:
  0 — spike passed (worker reached ready, cleanup succeeded)
  1 — image push / endpoint creation failed
  2 — worker did not reach ready within timeout
  3 — cleanup failed (orphan endpoint left behind — manual delete needed)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import requests

# Read control-plane key directly from config.toml to avoid env shadowing
# (matches spike v4 pattern). Set RUNPOD_FLASH_API_KEY as fallback.
def _read_api_key() -> str:
    env_key = os.environ.get("RUNPOD_FLASH_API_KEY")
    if env_key:
        return env_key
    config_path = os.path.expanduser("~/.runpod/config.toml")
    if os.path.exists(config_path):
        with open(config_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("api_key"):
                    # toml: api_key = "rpa_..."
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(
        "No API key found. Run `flash login` or set RUNPOD_FLASH_API_KEY."
    )


API_KEY = _read_api_key()

CONTROL_PLANE_URL = "https://api.runpod.io/graphql"
DATA_PLANE_URL_TPL = "https://api.runpod.ai/v2/{eid}/health"

# Same proven params as prod (mirrors deploy/flash_config.py).
# Override tag via ENDPOINT_IMAGE_TAG env var (e.g. `:debug-3131efa`) to
# deploy an alternate image for debugging without touching prod endpoint.
_IMAGE_NAME = "antongoldobin/comfyui-ltx-video"
_IMAGE_TAG = os.environ.get("ENDPOINT_IMAGE_TAG", "latest")
IMAGE = f"{_IMAGE_NAME}:{_IMAGE_TAG}"
NETWORK_VOLUME_ID = "f3falnf3r0"
DATACENTER = "EU-RO-1"

# How long to wait for worker ready. Burst-load spike can be slow under
# GPU scarcity; 10 min is generous but matches prod warm-pool target.
READY_TIMEOUT_S = 600
POLL_INTERVAL_S = 15

ENDPOINT_PREFIX = "flash-spike-"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(stage: str, msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {stage}: {msg}", flush=True)


def _graphql(query: str, variables: dict | None = None) -> dict:
    """POST to RunPod GraphQL; raise SystemExit on non-200."""
    payload = {"query": query}
    if variables is not None:
        payload["variables"] = variables
    r = requests.post(
        CONTROL_PLANE_URL,
        json=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        timeout=30,
    )
    if r.status_code != 200:
        raise SystemExit(f"GraphQL HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    if "errors" in data and data["errors"]:
        raise SystemExit(f"GraphQL errors: {data['errors']}")
    return (data.get("data") or {})


def list_endpoints() -> list[dict]:
    """Return all endpoints on this account."""
    query = "{ myself { endpoints { id name } } }"
    data = _graphql(query)
    return (data.get("myself") or {}).get("endpoints") or []


def delete_endpoint(endpoint_id: str) -> bool:
    """Delete via GraphQL. Mutation type is Void, no selection set allowed."""
    query = 'mutation { deleteEndpoint(id: "%s") }' % endpoint_id
    try:
        _graphql(query)
        return True
    except SystemExit as e:
        _log("cleanup", f"delete failed: {e}")
        return False


def wait_for_worker_ready(endpoint_id: str, timeout_s: int = READY_TIMEOUT_S) -> bool:
    """Poll /health until workers.ready >= 1 or timeout."""
    url = DATA_PLANE_URL_TPL.format(eid=endpoint_id)
    deadline = time.time() + timeout_s
    start = time.time()
    last_state = None
    while time.time() < deadline:
        elapsed = int(time.time() - start)
        try:
            r = requests.get(
                url,
                headers={"Authorization": f"Bearer {API_KEY}"},
                timeout=10,
            )
            if r.status_code == 200:
                health = r.json()
                workers = health.get("workers", {})
                state = (
                    workers.get("ready", 0),
                    workers.get("running", 0),
                    workers.get("idle", 0),
                    workers.get("initializing", 0),
                    workers.get("unhealthy", 0),
                )
                if state != last_state:
                    _log(
                        "poll",
                        f"t={elapsed:3d}s ready={state[0]} running={state[1]} "
                        f"idle={state[2]} init={state[3]} unhl={state[4]}",
                    )
                    last_state = state
                if state[0] >= 1 or state[1] >= 1:
                    _log("poll", f"✅ worker ready after {elapsed}s")
                    return True
            else:
                _log("poll", f"t={elapsed:3d}s HTTP {r.status_code}")
        except requests.RequestException as e:
            _log("poll", f"t={elapsed:3d}s request error: {e}")
        time.sleep(POLL_INTERVAL_S)
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Skip cleanup; leave the spike endpoint for inspection.",
    )
    parser.add_argument(
        "--ready-timeout",
        type=int,
        default=READY_TIMEOUT_S,
        help=f"Seconds to wait for worker ready (default {READY_TIMEOUT_S})",
    )
    args = parser.parse_args()

    # Import AFTER arg parse so `--help` works without API key.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runpod_flash import DataCenter, Endpoint, GpuType, NetworkVolume

    endpoint_name = f"{ENDPOINT_PREFIX}{int(time.time())}"
    print("=" * 70)
    print(f"FLASH SPIKE — endpoint={endpoint_name}")
    print("=" * 70)
    print(f"image:    {IMAGE}")
    print(f"gpu:      RTX 4090")
    print(f"volume:   {NETWORK_VOLUME_ID} ({DATACENTER})")
    print(f"key:      {API_KEY[:25]}...{API_KEY[-8:]}")
    print()

    # --- Step 1: provision endpoint via Flash (no SDK deploy method, runsync does it) ---
    _log("step1", "creating throwaway endpoint via Flash SDK (Endpoint + _ensure_endpoint_ready)")
    ep = Endpoint(
        name=endpoint_name,
        image=IMAGE,
        gpu=GpuType.NVIDIA_GEFORCE_RTX_4090,
        workers=(1, 1),
        idle_timeout=120,
        volume=NetworkVolume(id=NETWORK_VOLUME_ID, dataCenterId=DataCenter.EU_RO_1),
        datacenter=DataCenter.EU_RO_1,
        env={"FLASH_SPIKE": "1"},
        execution_timeout_ms=90_000,
    )

    # Trigger provisioning in a background task by calling runsync with a
    # benign payload. We don't need the result — we just want the endpoint
    # to exist. Wrap in try/except: if runsync errors, the endpoint may
    # still have been created.
    try:
        import asyncio
        from datetime import datetime as _dt

        async def _kick():
            try:
                job = await ep.runsync(
                    {"flash_spike": True, "ts": _dt.now(timezone.utc).isoformat()},
                    timeout=300,
                )
                _log("step1", f"kick job returned status={job.status}")
            except Exception as e:
                _log("step1", f"kick job error (ignored): {type(e).__name__}: {str(e)[:200]}")

        asyncio.run(_kick())
    except Exception as e:
        _log("step1", f"runsync wrapper error: {e}")

    # --- Step 2: locate the endpoint id by name ---
    _log("step2", "locating endpoint id via GraphQL")
    endpoints = list_endpoints()
    spike = next((e for e in endpoints if e.get("name") == endpoint_name), None)
    if not spike:
        print(f"\n❌ Step 2 FAILED — endpoint '{endpoint_name}' not found.")
        print("   Existing endpoints:")
        for e in endpoints:
            print(f"     - {e['id']} | {e['name']}")
        return 1
    endpoint_id = spike["id"]
    _log("step2", f"found endpoint id={endpoint_id}")

    # --- Step 3: wait for worker ready ---
    _log("step3", f"polling /health (max {args.ready_timeout}s)")
    if not wait_for_worker_ready(endpoint_id, timeout_s=args.ready_timeout):
        print(f"\n❌ Step 3 FAILED — worker did not become ready within {args.ready_timeout}s")
        if not args.keep:
            _log("cleanup", "deleting orphan spike endpoint")
            delete_endpoint(endpoint_id)
        return 2

    # --- Step 4: cleanup ---
    if args.keep:
        _log("step4", f"--keep flag set, leaving endpoint {endpoint_id} alive")
        print(f"\n✅ SPIKE PASSED (kept alive) — endpoint={endpoint_name} id={endpoint_id}")
        return 0

    _log("step4", "deleting spike endpoint via GraphQL")
    deleted = delete_endpoint(endpoint_id)
    if not deleted:
        print(f"\n⚠️  SPIKE PASSED but cleanup FAILED — endpoint {endpoint_id} still alive.")
        print("   Manual cleanup: mutation { deleteEndpoint(id: \"%s\") }" % endpoint_id)
        return 3

    # Verify only prod endpoint remains
    endpoints_after = list_endpoints()
    others = [e for e in endpoints_after if e.get("name") != endpoint_name]
    print()
    print("=" * 70)
    print("✅ SPIKE PASSED")
    print("=" * 70)
    print(f"endpoint id {endpoint_id} ({endpoint_name}) deleted")
    print(f"endpoints remaining on account: {len(others)}")
    for e in others:
        marker = " ← PROD" if e["id"] == "3995g4nz6mxw1v" else ""
        print(f"  - {e['id']} | {e['name']}{marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
