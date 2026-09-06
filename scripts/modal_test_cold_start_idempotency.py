"""
R.119 Шаг 0 — cold-start idempotency check.

QUESTION: When /api/run hits a cold container and the transport breaks at 150s
(307 redirect / 400 bad-method), does H3Generator().generate.remote() ACTUALLY
enter the function on the GPU container? Or is the redirect strictly queued
BEFORE the function entry, so retry creates a clean new job (no duplicate)?

TEST:
  1. MarkerTestGenerator.run() writes nonce+timestamp to a Modal Dict IMMEDIATELY
     on entry, BEFORE any sleep / work.
  2. /api/test-run proxy in serve() calls MarkerTestGenerator().run.remote(...)
     via asyncio.to_thread.
  3. Warm scenario: POST /api/test-run (sleep=5) → marker appears, 200 OK.
  4. Wait > 5s for scaledown.
  5. Cold scenario: POST /api/test-run (sleep=200, nonce='cold-test-X') →
     client transport breaks at ~150s (307/400).
  6. Wait full duration (250s).
  7. GET /api/markers → check if 'cold-test-X' marker exists.

INTERPRETATION:
  - Marker EXISTS → function was entered → retry creates duplicate GPU job
                    → MUST add nonce-based idempotency to generate() in Task #268.
  - Marker DOES NOT EXIST → function never started → retry is safe as-is.

App: comfyui-minimax-h3-r119-coldtest (separate from Step 0.5 app to isolate state).
"""
import asyncio
import json
import time

import modal

app = modal.App("comfyui-minimax-h3-r119-coldtest")

# Default Modal image lacks fastapi. Reuse the same debian_slim + fastapi.
shared_image = modal.Image.debian_slim(python_version="3.11").pip_install("fastapi")

# Modal Dict to track function entry timestamps + nonces. Shared across containers.
entry_markers = modal.Dict.from_name("r119-entry-markers", create_if_missing=True)


@app.cls(
    image=shared_image,
    cpu=1,
    memory=512,
    timeout=900,
    scaledown_window=5,              # matches prod autoscaler scaledown_window=5 (was container_idle_timeout in older Modal API)
    enable_memory_snapshot=False,    # disable snapshot — we WANT cold starts
)
class MarkerTestGenerator:
    @modal.enter()
    def setup(self):
        # Per-instance state. Reset on every cold start.
        self.local_entries = 0
        print("=== MarkerTestGenerator.setup() called ===")

    @modal.method()
    def run(self, sleep_seconds: int, nonce: str) -> dict:
        # === ENTRY MARKER ===
        # Written IMMEDIATELY, before any other work. If this Dict gets the
        # nonce, the function ran on the GPU container.
        marker = {
            "entered_at": time.time(),
            "sleep_seconds": sleep_seconds,
            "local_entries_after": self.local_entries + 1,
        }
        entry_markers[nonce] = marker
        self.local_entries += 1
        print(f"[ENTER] nonce={nonce} marker={marker}")

        # The "real work" — just sleep. No GPU/ComfyUI needed for this test.
        time.sleep(sleep_seconds)
        return {"nonce": nonce, "result": f"awake after {sleep_seconds}s", "marker": marker}


@app.function(
    image=shared_image,
    cpu=1,
    memory=512,
    timeout=900,
)
@modal.asgi_app()
def serve():
    from fastapi import FastAPI, Request

    web_app = FastAPI(title="R.119 Шаг 0 cold-start idempotency test")

    @web_app.post("/api/test-run")
    async def test_run(request: Request):
        body = await request.body()
        sleep_seconds = 5
        nonce = f"auto-{int(time.time()*1000)}"
        if body:
            try:
                payload = json.loads(body)
                sleep_seconds = int(payload.get("sleep", 5))
                nonce = payload.get("nonce", nonce)
            except Exception:
                pass

        # Mirror prod /api/run: call .remote() via asyncio.to_thread
        # so the FastAPI handler doesn't block on the long-running call.
        def call_remote():
            return MarkerTestGenerator().run.remote(sleep_seconds, nonce)

        t0 = time.time()
        try:
            result = await asyncio.to_thread(call_remote)
            elapsed = time.time() - t0
            return {
                "outcome": "success",
                "elapsed": round(elapsed, 2),
                "result": result,
            }
        except Exception as e:
            elapsed = time.time() - t0
            return {
                "outcome": "exception",
                "elapsed": round(elapsed, 2),
                "error_type": type(e).__name__,
                "error_msg": str(e)[:500],
            }

    @web_app.get("/api/markers")
    async def get_markers():
        # Read all markers from the Dict
        all_markers = dict(entry_markers.items())
        return {"count": len(all_markers), "markers": all_markers}

    @web_app.post("/api/markers/clear")
    async def clear_markers():
        # Clear all markers for a fresh test
        keys = list(entry_markers.keys())
        for k in keys:
            del entry_markers[k]
        return {"cleared": len(keys)}

    @web_app.get("/api/health")
    async def health():
        return {"ok": True}

    return web_app
