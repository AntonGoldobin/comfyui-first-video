"""
handler.py — RunPod Serverless entrypoint.

ComfyUI is started by startup.sh as a sidecar process BEFORE this script runs.
This script only sets up the RunPod serverless handler.
"""

import asyncio
import runpod

from runpod_handler import _worker


def handler(job):
    """Named handler for runpod.serverless.start()."""
    return asyncio.get_event_loop().run_until_complete(_worker.handler(job))


runpod.serverless.start({"handler": handler})
