"""
R.119 Step 0 — verify Modal .remote() timeout limit before main rewrite.

Hypothesis under test (from earlier code comments):
  ".remote() has 150s HTTP timeout cap" — R.148 observation
Test it empirically with a trivial sleep function.

Tests:
  A. .remote(60s)    — sanity
  B. .remote(200s)   — alleged 150s cap region
  C. .remote(400s)   — above default function timeout=300s
  D. .remote.aio(400s) — async alternative
  E. .spawn().get(400s) — handle-based alternative

Each test reports actual elapsed wall-clock + outcome.
If .remote() fails anywhere, alternatives are auto-tested.

App name: timeout-test-r119 (separate from prod comfyui-minimax-h3).

Usage:
  modal run scripts/modal_test_timeout.py
"""
import asyncio
import time

import modal

app = modal.App("timeout-test-r119")


@app.function(timeout=900)  # 15min cap to cover 550s worst case
def sleeper(n_seconds: int) -> str:
    """Sleep n_seconds, return a confirmation string. No GPU, minimal CPU."""
    time.sleep(n_seconds)
    return f"awake after {n_seconds}s"


def _fmt(elapsed: float, result: str) -> str:
    return f"elapsed={elapsed:7.2f}s  {result}"


def test_remote(n_seconds: int) -> tuple[float, str]:
    t0 = time.time()
    try:
        result = sleeper.remote(n_seconds)
        elapsed = time.time() - t0
        return elapsed, f"OK: {result!r}"
    except Exception as e:
        elapsed = time.time() - t0
        return elapsed, f"FAIL: {type(e).__name__}: {e}"


async def test_remote_aio(n_seconds: int) -> tuple[float, str]:
    t0 = time.time()
    try:
        result = await sleeper.remote.aio(n_seconds)
        elapsed = time.time() - t0
        return elapsed, f"OK: {result!r}"
    except Exception as e:
        elapsed = time.time() - t0
        return elapsed, f"FAIL: {type(e).__name__}: {e}"


def test_spawn(n_seconds: int) -> tuple[float, str]:
    t0 = time.time()
    try:
        call = sleeper.spawn(n_seconds)
        result = call.get()
        elapsed = time.time() - t0
        return elapsed, f"OK: {result!r}"
    except Exception as e:
        elapsed = time.time() - t0
        return elapsed, f"FAIL: {type(e).__name__}: {e}"


@app.local_entrypoint()
def main():
    print("=" * 60)
    print("R.119 Step 0 — Modal .remote() timeout verification")
    print("=" * 60)

    remote_works = True

    # ── A. sanity ─────────────────────────────────────────────────────
    print("\n[A] .remote(60s) — sanity")
    e, r = test_remote(60)
    print(f"    {_fmt(e, r)}")
    if "FAIL" in r:
        remote_works = False
        print("    !! sanity failed — aborting")

    # ── B. alleged 150s cap region ────────────────────────────────────
    print("\n[B] .remote(200s) — alleged 150s cap region")
    if remote_works:
        e, r = test_remote(200)
        print(f"    {_fmt(e, r)}")
        if "FAIL" in r:
            remote_works = False

    # ── C. above default function timeout=300s ────────────────────────
    print("\n[C] .remote(400s) — above default function timeout=300s")
    if remote_works:
        e, r = test_remote(400)
        print(f"    {_fmt(e, r)}")
        if "FAIL" in r:
            remote_works = False

    # ── D. async alternative ──────────────────────────────────────────
    print("\n[D] .remote.aio(400s) — async alternative")
    e, r = asyncio.run(test_remote_aio(400))
    print(f"    {_fmt(e, r)}")

    # ── E. handle-based alternative ────────────────────────────────────
    print("\n[E] .spawn().get(400s) — handle-based alternative")
    e, r = test_spawn(400)
    print(f"    {_fmt(e, r)}")

    print("\n" + "=" * 60)
    print("DONE.")
    print("=" * 60)
