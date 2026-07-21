"""
handler.py — RunPod Serverless entrypoint.

ComfyUI is started by base image startup.sh as a background process.
This script waits for ComfyUI to be fully ready BEFORE accepting jobs.
"""

import asyncio
import os
import requests
import time
import runpod

from runpod_handler import _worker

COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1:8188")
COMFY_API_AVAILABLE_MAX_RETRIES = int(os.environ.get("COMFY_API_AVAILABLE_MAX_RETRIES", 300))
COMFY_API_AVAILABLE_INTERVAL_MS = int(os.environ.get("COMFY_API_AVAILABLE_INTERVAL_MS", 1000))
COMFY_PID_FILE = os.environ.get("COMFY_PID_FILE", "/tmp/comfyui.pid")


def _is_comfyui_process_alive():
    """Check if ComfyUI process is still alive via PID file."""
    try:
        with open(COMFY_PID_FILE) as f:
            pid = int(f.read().strip())
        import os
        os.kill(pid, 0)
        return True
    except (FileNotFoundError, ProcessLookupError, ValueError):
        return False


def check_server(url, retries=0, delay_ms=50):
    """Wait for ComfyUI /system_stats endpoint to be reachable.

    Returns True if ComfyUI becomes reachable within retries.
    Returns False if max retries reached (ComfyUI failed to start).
    """
    delay_sec = max(0.001, delay_ms / 1000)
    log_every = max(1, int(10 / delay_sec)) if delay_sec < 10 else 1

    attempt = 0
    while True:
        # Check if ComfyUI process crashed
        process_alive = _is_comfyui_process_alive()
        if process_alive is False:
            print("ComfyUI process has exited unexpectedly.")
            return False

        # Check HTTP reachability
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"ComfyUI API is reachable after {attempt} attempts")
                return True
        except requests.RequestException:
            pass

        attempt += 1
        fallback_retries = retries if retries > 0 else 500

        # If no PID file and max retries reached, give up
        if process_alive is None and attempt >= fallback_retries:
            print(f"Max retries ({fallback_retries}) reached, giving up")
            return False

        if attempt % log_every == 0:
            print(f"Waiting for ComfyUI... ({attempt}/{fallback_retries})")

        time.sleep(delay_sec)


# BLOCK until ComfyUI is ready before accepting any jobs
comfy_url = f"http://{COMFY_HOST}/system_stats"
print(f"Waiting for ComfyUI at {comfy_url} ...")
print(f"Max retries: {COMFY_API_AVAILABLE_MAX_RETRIES}, interval: {COMFY_API_AVAILABLE_INTERVAL_MS}ms")

if not check_server(comfy_url, COMFY_API_AVAILABLE_MAX_RETRIES, COMFY_API_AVAILABLE_INTERVAL_MS):
    print("WARNING: ComfyUI not reachable after max retries, starting handler anyway")
else:
    print("ComfyUI is ready, starting handler")


async def handler(job):
    """Named handler for runpod.serverless.start()."""
    return await _worker.handler(job)


runpod.serverless.start({"handler": handler})
