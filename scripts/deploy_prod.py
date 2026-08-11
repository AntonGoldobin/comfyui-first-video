"""
deploy_prod.py — Production cutover script (phase 17, DoD-6).

Migrates the production endpoint `3995g4nz6mxw1v` (`hon_cyan_tiger`) from
manual Console redeploy to Flash SDK auto-deploy. After this:

  - Future updates: just `git push` → GH Actions builds → flash deploy runs.
  - Old endpoint id is destroyed and recreated (image is preserved on Docker Hub).
  - Worker pool becomes Flash-managed (1-3 warm).

WARNING: this destroys and recreates the endpoint. The endpoint ID may change.
Update Reelant-side `RUNPOD_ENDPOINT_ID` if the new id differs.

Pre-flight:
  - `flash login` has populated ~/.runpod/config.toml with control-plane key
  - OR env var RUNPOD_FLASH_API_KEY is set

Exit codes:
  0 — prod endpoint ready, workers healthy
  1 — endpoint recreation failed
  2 — worker did not become ready within timeout
  3 — old endpoint id unreachable (manual cleanup likely needed)
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from datetime import datetime, timezone

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deploy.flash_config import ep  # noqa: E402

PROD_ENDPOINT_NAME = "hon_cyan_tiger"
OLD_PROD_EID = "3995g4nz6mxw1v"  # the one we know about; new id will differ
READY_TIMEOUT_S = 600
POLL_INTERVAL_S = 15

DATA_PLANE_URL_TPL = "https://api.runpod.ai/v2/{eid}/health"
CONTROL_PLANE_URL = "https://api.runpod.io/graphql"


def _read_api_key() -> str:
    env_key = os.environ.get("RUNPOD_FLASH_API_KEY")
    if env_key:
        return env_key
    config_path = os.path.expanduser("~/.runpod/config.toml")
    with open(config_path) as f:
        for line in f:
            if line.strip().startswith("api_key"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("No API key. Run `flash login` or set RUNPOD_FLASH_API_KEY.")


def _log(stage: str, msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {stage}: {msg}", flush=True)


def _graphql(query: str) -> dict:
    r = requests.post(
        CONTROL_PLANE_URL,
        json={"query": query},
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_read_api_key()}",
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
    data = _graphql("{ myself { endpoints { id name } } }")
    return (data.get("myself") or {}).get("endpoints") or []


def delete_endpoint(endpoint_id: str) -> bool:
    query = f'mutation {{ deleteEndpoint(id: "{endpoint_id}") }}'
    try:
        _graphql(query)
        return True
    except SystemExit as e:
        _log("cleanup", f"delete {endpoint_id} failed: {e}")
        return False


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


async def trigger_deploy() -> None:
    """Force Flash to (re-)provision the endpoint by sending a kick job."""
    try:
        job = await ep.runsync(
            {"deploy_prod": True, "ts": datetime.now(timezone.utc).isoformat()},
            timeout=600,
        )
        _log("deploy", f"kick job returned status={job.status}")
    except Exception as e:
        _log("deploy", f"kick job warning (endpoint may still be ready): {type(e).__name__}: {e}")


def main() -> int:
    print("=" * 70)
    print(f"PROD CUTOVER — endpoint={PROD_ENDPOINT_NAME}")
    print("=" * 70)
    print(f"image:   {ep.image}")
    print(f"workers: min={ep.workers_min}, max={ep.workers_max}")
    print()

    # --- Pre-flight: existing endpoints ---
    _log("preflight", "listing endpoints")
    before = list_endpoints()
    pre_existing = [e for e in before if e["name"] == PROD_ENDPOINT_NAME]
    if pre_existing:
        pre_eid = pre_existing[0]["id"]
        _log("preflight", f"existing endpoint {PROD_ENDPOINT_NAME}={pre_eid}")

    # --- Step 1: delete existing endpoint (Flash will recreate) ---
    if pre_existing:
        _log("step1", f"deleting existing endpoint id={pre_existing[0]['id']}")
        if not delete_endpoint(pre_existing[0]["id"]):
            _log("step1", "FAILED to delete — aborting (existing endpoint kept)")
            return 3

    # --- Step 2: trigger Flash recreate via runsync ---
    _log("step2", "calling ep.runsync() to provision new endpoint")
    asyncio.run(trigger_deploy())

    # --- Step 3: locate new endpoint id ---
    _log("step3", "locating new endpoint id via GraphQL")
    after = list_endpoints()
    prod_after = next((e for e in after if e["name"] == PROD_ENDPOINT_NAME), None)
    if not prod_after:
        print(f"\n❌ Step 3 FAILED — endpoint '{PROD_ENDPOINT_NAME}' not found after deploy")
        print("   Existing endpoints now:")
        for e in after:
            print(f"     - {e['id']} | {e['name']}")
        return 1
    new_eid = prod_after["id"]
    _log("step3", f"new endpoint id={new_eid}")

    if new_eid == OLD_PROD_EID:
        _log("step3", "✅ new endpoint id matches old — Reelant integration unchanged")
    else:
        print()
        print("=" * 70)
        print(f"⚠️  ENDPOINT ID CHANGED")
        print("=" * 70)
        print(f"   old: {OLD_PROD_EID}")
        print(f"   new: {new_eid}")
        print(f"   → Update Reelant RUNPOD_ENDPOINT_ID")
        print()

    # --- Step 4: wait for worker ready ---
    _log("step4", f"polling /health (max {READY_TIMEOUT_S}s)")
    if not wait_for_worker_ready(new_eid, READY_TIMEOUT_S):
        print(f"\n❌ Step 4 FAILED — worker not ready within {READY_TIMEOUT_S}s")
        return 2

    # --- Summary ---
    print()
    print("=" * 70)
    print("✅ PROD CUTOVER PASSED")
    print("=" * 70)
    print(f"endpoint: {PROD_ENDPOINT_NAME} ({new_eid})")
    print(f"image:    {ep.image}")
    if new_eid != OLD_PROD_EID:
        print(f"⚠️  Update Reelant RUNPOD_ENDPOINT_ID: {new_eid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
