"""
R.119 Step 0.5 — verify Modal Web Function 150s HTTP timeout behavior.

Hypothesis under test:
  Modal Web Function endpoints (asgi_app, fastapi_endpoint, webhook) have a
  platform-level 150s HTTP request timeout. When the inner function takes
  longer, Modal returns HTTP 303 redirect to a "result URL" — and the client
  must follow it. Per Modal docs, up to 20 such redirects are allowed (~50 min).

  R.119 Step 0 debunked the 150s cap for `.remote()` Modal-internal RPC.
  This Step 0.5 verifies the SAME is true for the OUTER HTTP transport from
  worker → serve() → /api/run.

Tests:
  - /api/test-run blocks for 200 / 400 / 550s (worst case from R.148)
  - Hit with curl (baseline) AND Node + @nestjs/axios (real worker client)

App name: comfyui-minimax-h3-r119-test (separate from prod).
Usage:
  modal deploy scripts/modal_test_web_timeout.py
  # then hit https://anton722451--comfyui-minimax-h3-r119-test-serve.modal.run/api/test-run
"""
import asyncio
import time

import modal

app = modal.App("comfyui-minimax-h3-r119-test")

# Default Modal image lacks fastapi. Add a minimal debian_slim image
# with fastapi installed.
test_image = modal.Image.debian_slim(python_version="3.11").pip_install("fastapi")


@app.function(
    image=test_image,
    cpu=1,
    memory=512,
    timeout=900,  # function timeout — covers worst case 550s + redirect overhead
)
@modal.asgi_app()
def serve():
    """Minimal Web Function: POST /api/test-run blocks for N seconds, returns JSON.

    Mirrors the structure of prod serve() in modal_comfyui_minimax_h3.py:
    asgi_app decorator + FastAPI app + asyncio.to_thread wrapping a blocking
    call. The pattern we'll use in /api/run for R.119 generate() — if THIS
    passes the 150s test, the real /api/run will too.
    """
    from fastapi import FastAPI, Request

    web_app = FastAPI(title="R.119 Step 0.5 test endpoint")

    @web_app.post("/api/test-run")
    async def test_run(request: Request):
        body = await request.body()
        sleep_seconds = 200  # default
        if body:
            try:
                import json as _json

                payload = _json.loads(body)
                sleep_seconds = int(payload.get("sleep", 200))
            except Exception:
                pass

        def blocking_sleep():
            time.sleep(sleep_seconds)
            return f"awake after {sleep_seconds}s"

        t0 = time.time()
        result = await asyncio.to_thread(blocking_sleep)
        elapsed = time.time() - t0
        return {
            "result": result,
            "requestedSleep": sleep_seconds,
            "actualElapsed": round(elapsed, 2),
        }

    @web_app.get("/api/health")
    async def health():
        return {"ok": True}

    return web_app
