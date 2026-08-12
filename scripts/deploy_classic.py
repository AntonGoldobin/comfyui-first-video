"""
deploy_classic.py — Production cutover script (classic RunPod Serverless, not Flash).

Creates the prod endpoint `hon_cyan_tiger` via the classic RunPod GraphQL
API (saveEndpoint with inline serverless template) instead of Flash SDK.
This is the path required for the classic runpod SDK worker to work —
Flash runtime injects different env vars (RUNPOD_ENDPOINT_ID,
RUNPOD_FLASH_API_KEY) that don't satisfy the SDK's worker_state.WORKER_ID
and rp_job.JOB_GET_URL constants (see handler.py env-bridge workaround in
commit 9671f17 for details).

After this script:
  - Endpoint is created via classic API; classic env vars are injected
    natively by the runtime, no handler.py workaround strictly needed.
  - Worker pool (0, 3) ready, RTX 4090 at EU-RO-1, network volume mounted.

Pre-flight:
  - API key: `RUNPOD_FLASH_API_KEY` or `RAPA_TOKEN` env vars, or
    `~/.runpod/config.toml` with `api_key = ...`. Same control-plane key
    works for both classic and Flash (RunPod unified it).
  - Volume `f3falnf3r0` exists at EU-RO-1 (DO NOT DELETE).
  - Image `antongoldobin/comfyui-ltx-video:latest` exists on Docker Hub.

Exit codes:
  0 — endpoint ready, workers healthy
  1 — saveEndpoint failed
  2 — worker did not become ready within timeout
  3 — old endpoint id unreachable (manual cleanup likely needed)
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# Make the package root importable so `deploy.flash_config_classic` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deploy.flash_config_classic import cfg  # noqa: E402

PROD_ENDPOINT_NAME = cfg.name
OLD_PROD_EID = "f3l545i0ej27dz"  # the most recent endpoint id we know about
READY_TIMEOUT_S = 900
POLL_INTERVAL_S = 15

DATA_PLANE_URL_TPL = "https://api.runpod.ai/v2/{eid}/health"
CONTROL_PLANE_URL = "https://api.runpod.io/graphql"


# -----------------------------------------------------------------------------
# Auth
# -----------------------------------------------------------------------------

def _read_api_key() -> str:
    for env_name in ("RUNPOD_FLASH_API_KEY", "RAPA_TOKEN"):
        val = os.environ.get(env_name)
        if val:
            return val
    config_path = Path.home() / ".runpod" / "config.toml"
    if config_path.exists():
        for line in config_path.read_text().splitlines():
            if line.strip().startswith("api_key"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(
        "No API key. Set RUNPOD_FLASH_API_KEY env var or run `flash login`."
    )


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

def _log(stage: str, msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {stage}: {msg}", flush=True)


# -----------------------------------------------------------------------------
# GraphQL client
# -----------------------------------------------------------------------------

def _graphql(query: str, variables: dict | None = None) -> dict:
    key = _read_api_key()
    payload = {"query": query}
    if variables is not None:
        payload["variables"] = variables
    r = requests.post(
        CONTROL_PLANE_URL,
        json=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        timeout=60,
    )
    if r.status_code != 200:
        raise SystemExit(f"GraphQL HTTP {r.status_code}: {r.text[:400]}")
    body = r.json()
    if body.get("errors"):
        raise SystemExit(f"GraphQL errors: {body['errors']}")
    return body.get("data") or {}


# -----------------------------------------------------------------------------
# Schema: saveEndpoint, myself, deleteEndpoint
# -----------------------------------------------------------------------------

SAVE_ENDPOINT_MUTATION = """
mutation saveEndpoint($input: EndpointInput!) {
    saveEndpoint(input: $input) {
        id
        name
        gpuIds
        templateId
        workersMin
        workersMax
        idleTimeout
    }
}
"""

DELETE_ENDPOINT_MUTATION = """
mutation deleteEndpoint($id: String!) {
    deleteEndpoint(id: $id)
}
"""

DELETE_TEMPLATE_MUTATION = """
mutation deleteTemplate($name: String!) {
    deleteTemplate(templateName: $name)
}
"""

MYSELF_QUERY = """
{ myself { id } }
"""

LIST_ENDPOINTS_QUERY = """
{ myself { endpoints { id name } } }
"""


# -----------------------------------------------------------------------------
# Steps
# -----------------------------------------------------------------------------

def save_endpoint(endpoint_input: dict) -> str:
    """Create or update the classic serverless endpoint. Returns endpoint id.

    The template is passed INLINE — RunPod requires a serverless template
    (not the pod template you'd get from saveTemplate). Runtime creates
    a serverless template id from the inline fields.
    """
    _log("saveEndpoint", f"name={endpoint_input['name']} gpu={endpoint_input['gpuIds']}")
    data = _graphql(SAVE_ENDPOINT_MUTATION, {"input": endpoint_input})
    ep = (data.get("saveEndpoint") or {})
    eid = ep.get("id")
    if not eid:
        raise SystemExit(f"saveEndpoint returned no id: {ep}")
    _log("saveEndpoint", f"✅ endpoint id={eid} templateId={ep.get('templateId')}")
    return eid


def delete_endpoint(endpoint_id: str) -> bool:
    """Delete the existing endpoint by id (best-effort)."""
    try:
        _graphql(DELETE_ENDPOINT_MUTATION, {"id": endpoint_id})
        _log("cleanup", f"✅ deleted old endpoint {endpoint_id}")
        return True
    except SystemExit as e:
        _log("cleanup", f"delete {endpoint_id} failed: {e}")
        return False


def delete_template_by_name(name: str) -> bool:
    """Delete an orphaned serverless template by name (best-effort).

    RunPod rejects saveEndpoint with 'template names must be unique' if a
    previous endpoint was deleted but the template (named after the
    endpoint) lives on. Delete-by-name before re-create.
    """
    try:
        _graphql(DELETE_TEMPLATE_MUTATION, {"name": name})
        _log("cleanup", f"✅ deleted orphan template '{name}'")
        return True
    except SystemExit as e:
        _log("cleanup", f"delete template '{name}' failed: {e}")
        return False


def list_endpoints() -> list[dict]:
    data = _graphql(LIST_ENDPOINTS_QUERY)
    return (data.get("myself") or {}).get("endpoints") or []


def wait_for_worker_ready(endpoint_id: str, timeout_s: int) -> bool:
    """Poll /health on the data plane until workers.ready >= 1 or timeout."""
    url = DATA_PLANE_URL_TPL.format(eid=endpoint_id)
    key = _read_api_key()
    deadline = time.time() + timeout_s
    start = time.time()
    last_state = None
    while time.time() < deadline:
        elapsed = int(time.time() - start)
        try:
            r = requests.get(
                url,
                headers={"Authorization": f"Bearer {key}"},
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


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print(f"PROD CUTOVER (classic) — endpoint={PROD_ENDPOINT_NAME}")
    print("=" * 70)
    print(f"image:    {cfg.image}")
    print(f"gpu:      {cfg.gpu_id} @ {cfg.datacenter_id}")
    print(f"workers:  min={cfg.workers_min}, max={cfg.workers_max}")
    print(f"timeout:  idle={cfg.idle_timeout}s, exec={cfg.execution_timeout_ms}ms")
    print(f"volume:   {cfg.network_volume_id}")
    print()

    # --- Pre-flight: verify auth ---
    _log("preflight", "verifying API key via myself query")
    try:
        _graphql(MYSELF_QUERY)
    except SystemExit as e:
        _log("preflight", f"❌ auth failed: {e}")
        return 1
    _log("preflight", "✅ API key valid")

    # --- Pre-flight: list existing endpoints ---
    before = list_endpoints()
    pre_existing = [e for e in before if e["name"] == PROD_ENDPOINT_NAME]
    if pre_existing:
        _log("preflight", f"existing endpoint {PROD_ENDPOINT_NAME}={pre_existing[0]['id']}")

    # --- Step 1: delete existing endpoint (saves a re-create round trip) ---
    if pre_existing:
        _log("step1", f"deleting existing endpoint id={pre_existing[0]['id']}")
        if not delete_endpoint(pre_existing[0]["id"]):
            _log("step1", "warn: failed to delete — will use saveEndpoint upsert")

    # --- Step 1b: cleanup orphaned template (no endpoint referencing it) ---
    # RunPod rejects saveEndpoint with "template names must be unique" if a
    # previous endpoint was deleted but its template (named after the
    # endpoint, i.e. cfg.template_name) is still alive. Delete it now.
    _log("step1b", f"cleaning orphaned template '{cfg.template_name}' (if any)")
    delete_template_by_name(cfg.template_name)

    # --- Step 2: create endpoint with inline serverless template ---
    _log("step2", "creating endpoint via saveEndpoint (inline serverless template)")
    try:
        new_eid = save_endpoint(cfg.to_endpoint_input())
    except SystemExit as e:
        _log("step2", f"❌ FAILED: {e}")
        return 1

    # --- Step 3: id-change advisory ---
    if new_eid == OLD_PROD_EID:
        _log("step3", "✅ new endpoint id matches old — Reelant integration unchanged")
    else:
        print()
        print("=" * 70)
        print(f"⚠️  ENDPOINT ID CHANGED")
        print("=" * 70)
        print(f"   old: {OLD_PROD_EID}")
        print(f"   new: {new_eid}")
        print(f"   → Update Reelant RUNPOD_ENDPOINT_ID secret/env")
        print()

    # --- Step 4: wait for worker ready ---
    _log("step4", f"polling /health (max {READY_TIMEOUT_S}s)")
    if not wait_for_worker_ready(new_eid, READY_TIMEOUT_S):
        print(f"\n❌ Step 4 FAILED — worker not ready within {READY_TIMEOUT_S}s")
        return 2

    # --- Summary ---
    print()
    print("=" * 70)
    print("✅ PROD CUTOVER PASSED (classic)")
    print("=" * 70)
    print(f"endpoint: {PROD_ENDPOINT_NAME} ({new_eid})")
    print(f"image:    {cfg.image}")
    if new_eid != OLD_PROD_EID:
        print(f"⚠️  Update Reelant RUNPOD_ENDPOINT_ID: {new_eid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
