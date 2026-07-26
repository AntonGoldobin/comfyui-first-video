"""
flash_deploy_comfy.py — Alternative: Use Flash with pre-built ComfyUI image

NOTE: This approach uses Flash's image parameter to deploy a pre-built ComfyUI image.
The @Endpoint handler then proxies requests to the running ComfyUI server.

HOWEVER: This has limitations - Flash is designed for request-response functions,
not long-running servers. For production ComfyUI serverless, use the traditional
GitHub deployment approach instead.
"""

import asyncio
import os
import logging
from typing import Dict, Any

import httpx
from runpod_flash import Endpoint, GpuType, NetworkVolume

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ComfyUI server URL (running in same container via custom image)
COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")


@Endpoint(
    name="comfyui-proxy",
    gpu=GpuType.NVIDIA_GEFORCE_RTX_4090,
    image="runpod/worker-comfyui:5.8.4-base",  # Pre-built ComfyUI image
    volume=NetworkVolume(id="mbs1d3xwt0", name="reelant_volume", size=200),
    flashboot=True,
    workers=(0, 2),
    dependencies=["httpx"],
    env={
        "COMFY_URL": COMFY_URL,
    }
)
async def comfyui_proxy(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Proxy handler that submits workflows to local ComfyUI server.

    The custom image (runpod/worker-comfyui) is expected to:
    1. Start ComfyUI automatically on container boot
    2. Expose it at http://127.0.0.1:8188

    This handler then submits workflows and returns results.
    """
    workflow = job.get("input", {}).get("workflow")
    if not workflow:
        return {"status": "error", "error": "No workflow provided"}

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            # Submit workflow
            resp = await client.post(f"{COMFY_URL}/prompt", json={"prompt": workflow})
            if resp.status_code != 200:
                return {"status": "error", "error": f"Submit failed: {resp.text}"}

            prompt_id = resp.json().get("prompt_id")

            # Poll for completion
            for _ in range(300):  # 5 min timeout
                await asyncio.sleep(2)
                history_resp = await client.get(f"{COMFY_URL}/history/{prompt_id}")
                if history_resp.status_code == 200:
                    history = history_resp.json()
                    if prompt_id in history:
                        status = history[prompt_id].get("status", {})
                        if status.get("state") == "success":
                            return {
                                "status": "success",
                                "prompt_id": prompt_id,
                                "outputs": history[prompt_id].get("outputs", {})
                            }
                        if status.get("state") == "failed":
                            return {
                                "status": "error",
                                "error": f"Workflow failed: {status}"
                            }

            return {"status": "error", "error": "Workflow timed out"}

    except Exception as e:
        logger.exception("Error in comfyui_proxy")
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Flash — ComfyUI Proxy (Experimental)")
    logger.info("NOTE: For production, use GitHub deployment instead")
    logger.info("=" * 60)
    asyncio.run(comfyui_proxy())
