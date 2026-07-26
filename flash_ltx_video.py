"""
flash_ltx_video.py — RunPod Flash deployment for ComfyUI LTX Video

This script deploys ComfyUI with LTX Video to RunPod Serverless using Flash.
Run with: python flash_ltx_video.py

Flash automatically handles:
- GPU provisioning (RTX 4090/3090)
- Cold starts (Flash Boot enabled)
- Auto-scaling workers
- Network volume attachment for models
"""

import asyncio
import os
import sys
import json
import time
import subprocess
import logging
from typing import Dict, Any, Optional

from runpod_flash import Endpoint, GpuType, NetworkVolume

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Network volume — reuse existing volume mbs1d3xwt0
NETWORK_VOLUME = NetworkVolume(
    id="mbs1d3xwt0",
    name="reelant_volume",
    size=200  # GB, will reuse existing if id matches
)

# ComfyUI configuration
COMFY_PORT = int(os.environ.get("COMFY_PORT", "8188"))
COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1")
COMFY_URL = f"http://{COMFY_HOST}:{COMFY_PORT}"

# Network volume mount path
NETWORK_VOLUME_PATH = os.environ.get("NETWORK_VOLUME_PATH", "/runpod-volume")

# Model paths on network volume
MODELS_PATH = f"{NETWORK_VOLUME_PATH}/models"

# =============================================================================
# ComfyUI Management
# =============================================================================

def wait_for_comfy(url: str, timeout: int = 600) -> bool:
    """
    Wait for ComfyUI to be ready by polling /system_stats endpoint.

    Args:
        url: ComfyUI URL
        timeout: Max seconds to wait

    Returns:
        True if ComfyUI is ready, False otherwise
    """
    import requests

    start_time = time.time()
    max_retries = timeout // 2

    logger.info(f"Waiting for ComfyUI at {url}...")

    for attempt in range(max_retries):
        try:
            response = requests.get(f"{url}/system_stats", timeout=5)
            if response.status_code == 200:
                elapsed = time.time() - start_time
                logger.info(f"ComfyUI ready after {elapsed:.1f}s")
                return True
        except requests.exceptions.RequestException:
            pass

        # First 50 retries: 1s intervals, then 2s intervals
        sleep_time = 1 if attempt < 50 else 2
        time.sleep(sleep_time)

    logger.error(f"ComfyUI not ready after {timeout}s")
    return False


def start_comfyui() -> subprocess.Popen:
    """
    Start ComfyUI as a background process.

    Returns:
        Popen process handle
    """
    # ComfyUI startup command
    cmd = [
        "/opt/venv/bin/python",
        "-m", "comfyui_main",
        "--port", str(COMFY_PORT),
        "--listen", COMFY_HOST,
        "--disable-auto-launch",
    ]

    logger.info(f"Starting ComfyUI: {' '.join(cmd)}")

    # Set up environment with model paths
    env = os.environ.copy()
    env["COMFYUI_MODEL_PATH"] = MODELS_PATH

    # Start ComfyUI
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid  # New process group for clean shutdown
    )

    # Write PID to file
    with open("/tmp/comfyui.pid", "w") as f:
        f.write(str(process.pid))

    logger.info(f"ComfyUI started with PID {process.pid}")
    return process


def get_comfyui_pid() -> Optional[int]:
    """Read ComfyUI PID from file."""
    try:
        with open("/tmp/comfyui.pid") as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return None


def is_comfyui_running() -> bool:
    """Check if ComfyUI process is still running."""
    pid = get_comfyui_pid()
    if pid is None:
        return False
    try:
        os.kill(pid, 0)  # Signal 0 just checks existence
        return True
    except OSError:
        return False


# =============================================================================
# ComfyUI API Interaction
# =============================================================================

def submit_workflow(workflow: Dict[str, Any]) -> Optional[str]:
    """
    Submit a workflow to ComfyUI and return prompt_id.

    Args:
        workflow: ComfyUI workflow JSON

    Returns:
        prompt_id if successful, None otherwise
    """
    import requests

    try:
        response = requests.post(
            f"{COMFY_URL}/prompt",
            json={"prompt": workflow},
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            return result.get("prompt_id")

        logger.error(f"Submit failed: {response.status_code} {response.text}")
        return None

    except Exception as e:
        logger.error(f"Submit error: {e}")
        return None


def poll_history(prompt_id: str, timeout: int = 600) -> Optional[Dict[str, Any]]:
    """
    Poll ComfyUI history endpoint until workflow completes.

    Args:
        prompt_id: The prompt ID to poll for
        timeout: Max seconds to wait

    Returns:
        History dict if completed, None otherwise
    """
    import requests

    start_time = time.time()
    poll_interval = 2  # seconds

    while time.time() - start_time < timeout:
        try:
            response = requests.get(
                f"{COMFY_URL}/history/{prompt_id}",
                timeout=10
            )

            if response.status_code == 200:
                history = response.json()

                if prompt_id in history:
                    status = history[prompt_id].get("status", {})
                    if status.get("state") == "success":
                        logger.info(f"Workflow {prompt_id} completed")
                        return history[prompt_id]

                    if status.get("state") == "failed":
                        logger.error(f"Workflow {prompt_id} failed: {status}")
                        return None

        except Exception as e:
            logger.warning(f"Poll error: {e}")

        time.sleep(poll_interval)

    logger.error(f"Workflow {prompt_id} timed out after {timeout}s")
    return None


def find_output_files(prompt_id: str) -> list:
    """
    Find output video/image files from completed workflow.

    Returns:
        List of file paths
    """
    import os

    output_dirs = ["/tmp/comfyui/output", "/tmp/comfyui"]
    found_files = []

    for odir in output_dirs:
        if not os.path.exists(odir):
            continue

        # Check main output dir
        for fname in os.listdir(odir):
            if fname.endswith(('.mp4', '.webm', '.png', '.jpg', '.jpeg', '.gif')):
                found_files.append(os.path.join(odir, fname))

        # Check subdir by prompt_id
        prompt_dir = os.path.join(odir, prompt_id)
        if os.path.exists(prompt_dir):
            for fname in os.listdir(prompt_dir):
                if fname.endswith(('.mp4', '.webm', '.png', '.jpg', '.jpeg', '.gif')):
                    found_files.append(os.path.join(prompt_dir, fname))

    return found_files


# =============================================================================
# S3 Upload
# =============================================================================

def upload_to_s3(file_path: str, s3_key: str) -> Optional[str]:
    """
    Upload file to S3 and return presigned URL.

    Args:
        file_path: Local file path
        s3_key: S3 object key

    Returns:
        Presigned URL or None
    """
    import boto3
    from botocore.config import Config as BotoConfig

    # Get S3 config from environment
    bucket = os.environ.get('S3_BUCKET')
    access_key = os.environ.get('S3_ACCESS_KEY_ID')
    secret_key = os.environ.get('S3_SECRET_ACCESS_KEY')
    endpoint = os.environ.get('S3_ENDPOINT_URL')

    if not all([bucket, access_key, secret_key, endpoint]):
        logger.warning("S3 not configured, returning local path")
        return f"file://{file_path}"

    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            endpoint_url=endpoint,
            config=BotoConfig(signature_version='s3v4')
        )

        s3_client.upload_file(file_path, bucket, s3_key)

        # Generate presigned URL (7 days)
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': s3_key},
            ExpiresIn=604800
        )

        logger.info(f"Uploaded to S3: {s3_key}")
        return url

    except Exception as e:
        logger.error(f"S3 upload error: {e}")
        return None


# =============================================================================
# Flash Endpoint Handler
# =============================================================================

@Endpoint(
    name="comfyui-ltx-video",
    gpu=GpuType.NVIDIA_GEFORCE_RTX_4090,
    volume=NETWORK_VOLUME,
    flashboot=True,
    dependencies=[
        "torch",
        "transformers",
        "huggingface-hub",
        "opencv-python",
        "imageio-ffmpeg",
        "requests",
        "httpx",
        "boto3",
    ],
    env={
        "COMFY_PORT": str(COMFY_PORT),
        "NETWORK_VOLUME_PATH": NETWORK_VOLUME_PATH,
    }
)
async def comfyui_ltx_handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flash endpoint handler for ComfyUI LTX Video generation.

    This function runs on RunPod serverless GPU.

    Args:
        job: Input dict with:
            - workflow: ComfyUI workflow JSON
            - images: Optional dict of {node_id: base64_image_data}

    Returns:
        Dict with:
            - status: "success" or "error"
            - video_url: S3 presigned URL for generated video
            - images: List of output image URLs
            - error: Error message if failed
    """
    global _comfy_process, _comfy_ready

    # Lazy start ComfyUI on first job
    if not is_comfyui_running():
        logger.info("Starting ComfyUI for first job...")
        _comfy_process = start_comfyui()

        if not wait_for_comfy(COMFY_URL, timeout=600):
            return {
                "status": "error",
                "error": "ComfyUI failed to start within timeout"
            }

        _comfy_ready = True
        logger.info("ComfyUI ready, processing job")

    # Extract job input
    workflow = job.get("input", {}).get("workflow")
    if not workflow:
        return {
            "status": "error",
            "error": "No workflow provided in job input"
        }

    # Submit workflow
    prompt_id = submit_workflow(workflow)
    if not prompt_id:
        return {
            "status": "error",
            "error": "Failed to submit workflow to ComfyUI"
        }

    # Poll for completion
    result = poll_history(prompt_id, timeout=600)
    if not result:
        return {
            "status": "error",
            "prompt_id": prompt_id,
            "error": "Workflow timed out or failed"
        }

    # Find output files
    output_files = find_output_files(prompt_id)

    # Upload to S3
    output_urls = []
    for fpath in output_files:
        fname = os.path.basename(fpath)
        s3_key = f"comfy-outputs/{prompt_id}/{fname}"
        url = upload_to_s3(fpath, s3_key)
        if url:
            output_urls.append({"filename": fname, "url": url})
        else:
            output_urls.append({"filename": fname, "url": f"file://{fpath}"})

    return {
        "status": "success",
        "prompt_id": prompt_id,
        "outputs": output_urls
    }


# Global state
_comfy_process: Optional[subprocess.Popen] = None
_comfy_ready = False


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("RunPod Flash — ComfyUI LTX Video Serverless")
    logger.info("=" * 60)
    logger.info(f"Network Volume: {NETWORK_VOLUME_PATH}")
    logger.info(f"Models Path: {MODELS_PATH}")
    logger.info(f"Volume ID: {NETWORK_VOLUME.id}")
    logger.info("=" * 60)

    # Run the Flash endpoint
    # This will:
    # 1. Connect to RunPod with API key
    # 2. Provision GPU if needed
    # 3. Mount network volume
    # 4. Start the endpoint handler
    asyncio.run(comfyui_ltx_handler())
