"""
e2e_real_workflow.py — End-to-end test of the Reelant → RunPod → ComfyUI pipeline.

Submits api-workflow-minimal.json (LTX t2v, 17 nodes, VHS_VideoCombine)
to the prod endpoint and polls until COMPLETED. Verifies:

  1. worker accepted the job (no 4xx on submit)
  2. handler dispatched to ComfyUI (/prompt accepted)
  3. ComfyUI executed the workflow (status.completed)
  4. outputs were found in /comfyui/output/
  5. S3 upload produced presigned URLs in output.images[*].data

Usage:
    ENDPOINT_ID=1beu8wyg1wrcj9 API_KEY=rpa_... python scripts/e2e_real_workflow.py

Exit codes:
    0 — pipeline verified end-to-end (video uploaded, URL returned)
    1 — submit failed
    2 — handler error / FAILED status
    3 — handler returned but no video URL (S3 disabled? bug?)
    4 — timeout (worker never picked up the job — throttling?)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_PATH = REPO_ROOT / "api-workflow-minimal.json"

API_BASE = "https://api.runpod.ai/v2"
DATA_PLANE_KEY = os.environ["API_KEY"]
ENDPOINT_ID = os.environ["ENDPOINT_ID"]

POLL_TIMEOUT_S = int(os.environ.get("POLL_TIMEOUT_S", "900"))  # 15 min
POLL_INTERVAL_S = 5


def submit() -> str:
    workflow = json.loads(WORKFLOW_PATH.read_text())
    payload = {
        "input": {
            "workflow": workflow,
            "images": [],
            # No s3Config → handler uses env-injected S3 (Golden Antelope)
        },
    }
    url = f"{API_BASE}/{ENDPOINT_ID}/run"
    print(f"submitting workflow with {len(workflow)} nodes to {ENDPOINT_ID}")
    r = requests.post(
        url,
        json=payload,
        headers={"Authorization": f"Bearer {DATA_PLANE_KEY}"},
        timeout=30,
    )
    if r.status_code != 200:
        print(f"❌ submit HTTP {r.status_code}: {r.text[:400]}")
        sys.exit(1)
    jid = r.json().get("id")
    if not jid:
        print(f"❌ submit returned no job id: {r.text[:400]}")
        sys.exit(1)
    print(f"✅ job submitted: {jid}")
    return jid


def poll(jid: str) -> dict[str, Any]:
    url = f"{API_BASE}/{ENDPOINT_ID}/status/{jid}"
    deadline = time.time() + POLL_TIMEOUT_S
    start = time.time()
    last_status = None
    while time.time() < deadline:
        r = requests.get(
            url,
            headers={"Authorization": f"Bearer {DATA_PLANE_KEY}"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        status = data.get("status")
        elapsed = int(time.time() - start)
        if status != last_status:
            print(f"  t={elapsed:4d}s  status={status}")
            last_status = status
        if status == "COMPLETED":
            return data
        if status == "FAILED":
            return data
        if status == "CANCELLED":
            return data
        time.sleep(POLL_INTERVAL_S)
    print(f"❌ timeout after {POLL_TIMEOUT_S}s — worker never picked up job")
    sys.exit(4)


def verify_output(data: dict[str, Any]) -> None:
    status = data.get("status")
    if status == "FAILED":
        err = data.get("error") or data.get("output") or {}
        print(f"❌ job FAILED: {json.dumps(err, indent=2)[:800]}")
        sys.exit(2)
    if status == "CANCELLED":
        print(f"❌ job CANCELLED: {data}")
        sys.exit(2)
    if status != "COMPLETED":
        print(f"❌ unexpected status: {status}")
        sys.exit(2)

    output = data.get("output") or {}
    images = output.get("images") or []
    if not images:
        print(f"❌ output.images is empty (handler returned no files)")
        print(f"   full output: {json.dumps(output, indent=2)[:600]}")
        sys.exit(3)

    print(f"\n✅ handler returned {len(images)} file(s):")
    for i, img in enumerate(images):
        kind = img.get("type")
        data_field = img.get("data", "")
        if kind == "s3_url" and isinstance(data_field, str) and data_field.startswith("http"):
            preview = data_field[:120] + ("..." if len(data_field) > 120 else "")
            print(f"  [{i}] {img.get('filename'):30s}  type=s3_url  url={preview}")
        elif kind == "local":
            print(f"  [{i}] {img.get('filename'):30s}  type=LOCAL    data={data_field[:80]}  ⚠️  S3 disabled?")
        else:
            print(f"  [{i}] {img.get('filename'):30s}  type={kind}  data={data_field[:80]}")

    s3_urls = [i for i in images if i.get("type") == "s3_url" and isinstance(i.get("data"), str) and i["data"].startswith("http")]
    if not s3_urls:
        print(f"\n❌ no s3_url returned — handler did not upload. Check S3 env in endpoint config.")
        sys.exit(3)

    print(f"\n✅ END-TO-END PIPELINE VERIFIED")
    print(f"   • job submitted to RunPod: ✅")
    print(f"   • handler dispatched to ComfyUI: ✅ (workflow ran)")
    print(f"   • outputs found: ✅ ({len(images)} file(s))")
    print(f"   • S3 upload: ✅ ({len(s3_urls)} presigned URL(s))")


def main() -> int:
    print("=" * 70)
    print(f"Reelant → RunPod → ComfyUI E2E (endpoint={ENDPOINT_ID})")
    print(f"workflow: {WORKFLOW_PATH.name}")
    print(f"poll timeout: {POLL_TIMEOUT_S}s")
    print("=" * 70)
    if not WORKFLOW_PATH.exists():
        print(f"❌ workflow file not found: {WORKFLOW_PATH}")
        return 1
    jid = submit()
    result = poll(jid)
    verify_output(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
