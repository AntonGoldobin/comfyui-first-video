"""
flash_deploy_comfy.py — Deploy ComfyUI via Flash SDK

Правильный способ: используем Endpoint(image=...) как ОБЪЕКТ, не декоратор.
Flash сам создаст/обновит serverless endpoint с указанным Docker образом.
"""

import asyncio
import os
import logging
from typing import Dict, Any

from runpod_flash import Endpoint, GpuType, NetworkVolume

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Volume ID правильный: ugiamecmm0
NETWORK_VOLUME = NetworkVolume(
    id="ugiamecmm0",
    name="reelant_volume",
    size=200
)

# Создаём endpoint с кастомным Docker образом
comfy_endpoint = Endpoint(
    name="comfyui-ltx-worker",
    image="runpod/worker-comfyui:5.8.4-base",  # Базовый ComfyUI образ
    gpu=GpuType.NVIDIA_GEFORCE_RTX_4090,
    workers=(0, 2),  # min, max workers
    volume=NETWORK_VOLUME,
    flashboot=True,
    env={
        "MODEL_PATH": "/runpod-volume/models",
    }
)

# =============================================================================
# Job Submission
# =============================================================================

async def submit_workflow(workflow: Dict[str, Any]) -> Dict[str, Any]:
    """
    Отправляет workflow в ComfyUI endpoint.
    """
    logger.info("Submitting workflow to ComfyUI...")

    try:
        job = await comfy_endpoint.run({
            "input": {
                "workflow": workflow
            }
        })

        logger.info(f"Job submitted: {job.id}")
        await job.wait()
        logger.info(f"Job completed: {job.output}")
        return job.output

    except Exception as e:
        logger.exception("Error submitting workflow")
        return {"status": "error", "error": str(e)}


# =============================================================================
# Main
# =============================================================================

async def main():
    """Пример использования."""
    example_workflow = {
        "3": {
            "inputs": {
                "model": "ltx2310eros_v1.safetensors",
                "width": 848,
                "height": 480,
                "video_length": 33,
                "prompt": "a beautiful landscape",
                "negative_prompt": "blurry, low quality",
                "seed": 42,
                "steps": 20,
                "cfg": 1.0,
            },
            "class_type": "LTXVideoPipe"
        }
    }

    result = await submit_workflow(example_workflow)
    print(f"Result: {result}")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Flash — ComfyUI LTX Video Deployment")
    logger.info(f"Volume: ugiamecmm0")
    logger.info("=" * 60)

    asyncio.run(main())
