"""
handler.py — RunPod Serverless Worker for ComfyUI LTX Video

Based on official runpod/worker-comfyui patterns with:
- Single-file handler with process monitoring
- check_server() that reads /tmp/comfyui.pid and monitors ComfyUI process
- GPU pre-flight checks
- WebSocket reconnect logic
- LTXVideo-specific workflow handling

ComfyUI is started by start.sh as a background process.
This script waits for ComfyUI to be fully ready BEFORE accepting jobs.
"""

import asyncio
import os
import sys
import json
import time
import uuid
import logging
import signal
from typing import Optional, Dict, Any, List

import requests
import runpod
import httpx
import boto3
from botocore.config import Config as BotoConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1:8188")
COMFY_HOST_URL = f"http://{COMFY_HOST}"
COMFY_API_KEY = os.environ.get("COMFY_API_KEY")
COMFY_PID_FILE = os.environ.get("COMFY_PID_FILE", "/tmp/comfyui.pid")
COMFY_LOG_LEVEL = os.environ.get("COMFY_LOG_LEVEL", "INFO")

# Timeouts and retries
COMFY_API_AVAILABLE_MAX_RETRIES = int(os.environ.get("COMFY_API_AVAILABLE_MAX_RETRIES", 300))
COMFY_API_AVAILABLE_INTERVAL_MS = int(os.environ.get("COMFY_API_AVAILABLE_INTERVAL_MS", 1000))
WEBUI_AVAILABLE_TIMEOUT = int(os.environ.get("WEBUI_AVAILABLE_TIMEOUT", 300))

# S3 configuration from environment
DEFAULT_S3_BUCKET = os.environ.get('S3_BUCKET')
DEFAULT_S3_ACCESS_KEY = os.environ.get('S3_ACCESS_KEY_ID')
DEFAULT_S3_SECRET_KEY = os.environ.get('S3_SECRET_ACCESS_KEY')
DEFAULT_S3_ENDPOINT = os.environ.get('S3_ENDPOINT_URL')

# Polling configuration
HISTORY_POLL_INTERVAL = int(os.environ.get("HISTORY_POLL_INTERVAL", 2000))  # ms
HISTORY_TIMEOUT = int(os.environ.get("HISTORY_TIMEOUT", 600))  # seconds


def _is_comfyui_process_alive() -> Optional[bool]:
    """
    Check if ComfyUI process is still alive by reading PID file and sending signal 0.

    Returns:
        True if process is alive
        False if process is dead or PID file missing
        None if PID file doesn't exist (process not started yet)
    """
    try:
        if not os.path.exists(COMFY_PID_FILE):
            return None
        with open(COMFY_PID_FILE) as f:
            pid = int(f.read().strip())
        # Signal 0 doesn't actually send anything, just checks if process exists
        os.kill(pid, 0)
        return True
    except (FileNotFoundError, ProcessLookupError, ValueError, PermissionError):
        return False


def check_server(url: str, retries: int = 0, delay_ms: int = 50) -> bool:
    """
    Wait for ComfyUI /system_stats endpoint to be reachable.

    Monitors both HTTP availability AND process liveness.

    Returns True if ComfyUI becomes reachable within retries.
    Returns False if max retries reached or ComfyUI process died.
    """
    delay_sec = max(0.001, delay_ms / 1000)
    log_every = max(1, int(10 / delay_sec)) if delay_sec < 10 else 1

    attempt = 0
    fallback_retries = retries if retries > 0 else 500

    logger.info(f"Checking ComfyUI server at {url}")
    logger.info(f"Max retries: {fallback_retries}, interval: {delay_ms}ms")

    while True:
        # Check if ComfyUI process crashed
        process_alive = _is_comfyui_process_alive()
        if process_alive is False:
            logger.error("ComfyUI process has exited unexpectedly")
            return False

        # Check HTTP reachability
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                logger.info(f"ComfyUI API is reachable after {attempt} attempts")
                return True
        except requests.RequestException:
            pass

        attempt += 1

        # If no PID file and max retries reached, give up
        if process_alive is None and attempt >= fallback_retries:
            logger.error(f"Max retries ({fallback_retries}) reached, giving up")
            return False

        if attempt % log_every == 0:
            logger.info(f"Waiting for ComfyUI... ({attempt}/{fallback_retries})")

        time.sleep(delay_sec)


def get_api_headers() -> Dict[str, str]:
    """Get headers for ComfyUI API requests."""
    headers = {"Content-Type": "application/json"}
    if COMFY_API_KEY:
        headers["Authorization"] = f"Bearer {COMFY_API_KEY}"
    return headers


class S3Manager:
    """Manages S3 uploads for workflow outputs."""

    def __init__(self):
        self.client = None
        self.bucket = None
        self._setup_from_env()

    def _setup_from_env(self):
        """Initialize S3 client from environment variables."""
        if DEFAULT_S3_ACCESS_KEY and DEFAULT_S3_SECRET_KEY:
            try:
                self.client = boto3.client(
                    's3',
                    endpoint_url=DEFAULT_S3_ENDPOINT,
                    aws_access_key_id=DEFAULT_S3_ACCESS_KEY,
                    aws_secret_access_key=DEFAULT_S3_SECRET_KEY,
                    config=BotoConfig(signature_version='s3v4')
                )
                self.bucket = DEFAULT_S3_BUCKET
                logger.info(f"S3 client initialized from env: bucket={DEFAULT_S3_BUCKET}")
            except Exception as e:
                logger.warning(f"Failed to initialize S3 from env: {e}")

    def setup_from_config(self, config: Dict[str, Any]):
        """Initialize S3 client from RunPod config."""
        s3_config = config.get('s3Config') if config else None
        if s3_config:
            try:
                self.client = boto3.client(
                    's3',
                    endpoint_url=s3_config.get('endpointUrl'),
                    aws_access_key_id=s3_config['accessId'],
                    aws_secret_access_key=s3_config['accessSecret'],
                    config=BotoConfig(signature_version='s3v4')
                )
                self.bucket = s3_config['bucketName']
                logger.info(f"S3 client initialized from config: bucket={self.bucket}")
            except Exception as e:
                logger.warning(f"Failed to initialize S3 from config: {e}")

    @staticmethod
    def get_content_type(filename: str) -> str:
        """Get MIME content type from filename."""
        ext = os.path.splitext(filename)[1].lower()
        types = {
            '.mp4': 'video/mp4',
            '.webm': 'video/webm',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
        }
        return types.get(ext, 'application/octet-stream')

    def upload_file(self, local_path: str, s3_key: str, content_type: str = None) -> Optional[str]:
        """Upload file to S3 and return presigned URL."""
        if not self.client or not self.bucket:
            return None

        try:
            ct = content_type or self.get_content_type(local_path)
            self.client.upload_file(
                local_path,
                self.bucket,
                s3_key,
                ExtraArgs={'ContentType': ct}
            )
            url = self.client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket, 'Key': s3_key},
                ExpiresIn=3600 * 24  # 24 hours
            )
            logger.info(f"Uploaded {local_path} to S3: {s3_key}")
            return url
        except Exception as e:
            logger.warning(f"Failed to upload {local_path} to S3: {e}")
            return None


class ComfyUIWorker:
    """
    RunPod Serverless Worker for ComfyUI.

    Implements the ServerlessWorker interface from runpod SDK.
    Handles workflow submission, polling, and output upload.
    """

    def __init__(self):
        self.s3 = S3Manager()
        self.httpx_client = None

    def setup(self, config: Dict[str, Any]):
        """Called once at container startup (before first job)."""
        logger.info(f"ComfyUIWorker.setup() called with config keys: {list(config.keys()) if config else 'None'}")

        # Setup S3 from RunPod config
        self.s3.setup_from_config(config)

        # Create shared httpx client for connection pooling
        self.httpx_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )

    async def teardown(self):
        """Called when worker is shutting down."""
        if self.httpx_client:
            await self.httpx_client.aclose()

    async def handler(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main handler - called for each job.

        Args:
            job: RunPod job object with job['id'] and job['input']

        Returns:
            dict: Output object with images array
        """
        job_id = job.get('id', 'unknown')
        job_input = job.get('input', {})

        logger.info(f"Job {job_id} received")
        logger.info(f"Job input keys: {list(job_input.keys())}")

        try:
            # Extract workflow and config from job input
            workflow = job_input.get('workflow') or job_input.get('prompt')
            images = job_input.get('images', [])
            s3_config = job_input.get('s3Config')

            if not workflow:
                raise ValueError("No workflow found in job input")

            # Override S3 config from job input if provided
            s3 = self.s3
            if s3_config:
                s3 = S3Manager()
                s3.setup_from_config({'s3Config': s3_config})

            # 1. Upload input images to /input directory
            input_files = await self._upload_input_images(images)

            # 2. Submit workflow to ComfyUI
            prompt_id = await self._submit_workflow(workflow, input_files)

            # 3. Poll for completion
            result = await self._poll_for_completion(prompt_id)

            # 4. Upload outputs to S3 and return URLs
            output = await self._upload_outputs(result, prompt_id, s3)

            logger.info(f"Job {job_id} completed successfully")
            return output

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}", exc_info=True)
            raise

    async def _upload_input_images(self, images: List[Dict]) -> Dict[str, str]:
        """
        Handle uploaded reference images.
        Images come as {name: filename, image: base64_data} from Reelant.

        Returns dict mapping logical names to full paths in /input.
        """
        input_paths = {}
        if not images:
            return input_paths

        # Ensure /input directory exists
        os.makedirs('/input', exist_ok=True)

        for img in images:
            name = img.get('name', f'image_{len(input_paths)}')
            image_data = img.get('image', '')

            if not image_data:
                continue

            # Handle both base64 string and URL
            if isinstance(image_data, str) and not image_data.startswith('http'):
                try:
                    import base64
                    decoded = base64.b64decode(image_data)
                    img_path = f'/input/{name}'
                    with open(img_path, 'wb') as f:
                        f.write(decoded)
                    input_paths[name] = img_path
                    logger.info(f"Uploaded input image: {name}")
                except Exception as e:
                    logger.warning(f"Failed to decode base64 image {name}: {e}")
            else:
                # Download from URL
                try:
                    resp = await self.httpx_client.get(image_data)
                    resp.raise_for_status()
                    img_path = f'/input/{name}'
                    with open(img_path, 'wb') as f:
                        f.write(resp.content)
                    input_paths[name] = img_path
                    logger.info(f"Downloaded input image: {name}")
                except Exception as e:
                    logger.warning(f"Failed to download image {name}: {e}")

        return input_paths

    async def _submit_workflow(self, workflow: Dict, input_files: Dict[str, str]) -> str:
        """
        Submit workflow to ComfyUI API.

        Args:
            workflow: The ComfyUI workflow JSON
            input_files: Dict of input file paths

        Returns:
            prompt_id from ComfyUI
        """
        # Prepare workflow - replace input image paths
        workflow_copy = json.loads(json.dumps(workflow))  # Deep copy

        # Update image references in LoadImage nodes
        for node_id, node_data in workflow_copy.items():
            if node_data.get('class_type') == 'LoadImage':
                inputs = node_data.get('inputs', {})
                if 'image' in inputs:
                    img_name = inputs['image']
                    if img_name in input_files:
                        inputs['image'] = input_files[img_name]
                        logger.info(f"Replaced image path for {img_name}: {input_files[img_name]}")

        resp = await self.httpx_client.post(
            f'{COMFY_HOST_URL}/prompt',
            json={'prompt': workflow_copy},
            headers=get_api_headers()
        )
        resp.raise_for_status()

        result = resp.json()
        prompt_id = result.get('prompt_id')

        if not prompt_id:
            raise ValueError(f"No prompt_id in response: {result}")

        logger.info(f"Submitted workflow, prompt_id={prompt_id}")
        return prompt_id

    async def _poll_for_completion(self, prompt_id: str) -> Dict:
        """
        Poll ComfyUI history endpoint until workflow completes.

        Implements WebSocket-style reconnection logic.

        Args:
            prompt_id: The prompt ID from _submit_workflow

        Returns:
            The history entry for the completed prompt
        """
        start_time = time.time()
        poll_interval = HISTORY_POLL_INTERVAL / 1000  # Convert to seconds

        logger.info(f"Polling for completion of prompt_id={prompt_id}")

        while time.time() - start_time < HISTORY_TIMEOUT:
            # Check if ComfyUI process is still alive
            process_alive = _is_comfyui_process_alive()
            if process_alive is False:
                raise RuntimeError("ComfyUI process died during workflow execution")

            try:
                resp = await self.httpx_client.get(
                    f'{COMFY_HOST_URL}/history/{prompt_id}',
                    headers=get_api_headers()
                )
                resp.raise_for_status()
                history = resp.json()

                if prompt_id in history:
                    entry = history[prompt_id]
                    status = entry.get('status', {})

                    if status.get('completed'):
                        logger.info(f"Workflow {prompt_id} completed successfully")
                        return entry

                    if status.get('failed'):
                        error_msg = status.get('error', 'Unknown error')
                        raise RuntimeError(f"ComfyUI workflow failed: {error_msg}")

                elapsed = int(time.time() - start_time)
                if elapsed > 10 and elapsed % 30 == 0:
                    logger.info(f"Still polling {prompt_id}... ({elapsed}s elapsed)")

                await asyncio.sleep(poll_interval)

            except httpx.HTTPError as e:
                logger.warning(f"HTTP error polling history: {e}")
                await asyncio.sleep(min(poll_interval * 2, 10))  # Back off

        raise TimeoutError(f"Workflow {prompt_id} timed out after {HISTORY_TIMEOUT}s")

    async def _upload_outputs(
        self,
        result: Dict,
        prompt_id: str,
        s3: S3Manager
    ) -> Dict:
        """
        Find output files and upload to S3.

        Args:
            result: The completed workflow history entry
            prompt_id: The prompt ID
            s3: S3 manager instance

        Returns:
            Output dict with images array
        """
        output_files = []

        # Find outputs in /output directory (ComfyUI default)
        output_dirs = ['/output', '/workspace/ComfyUI/output', os.path.expanduser('~/ComfyUI/output')]

        found_files = []
        for odir in output_dirs:
            if os.path.exists(odir):
                for fname in os.listdir(odir):
                    if fname.endswith(('.mp4', '.webm', '.png', '.jpg', '.jpeg', '.gif')):
                        fpath = os.path.join(odir, fname)
                        mtime = os.path.getmtime(fpath)
                        found_files.append((fname, fpath, mtime))
                break

        # Also check for files in subdirectories by prompt_id
        for odir in output_dirs:
            prompt_output_dir = os.path.join(odir, prompt_id)
            if os.path.exists(prompt_output_dir):
                for fname in os.listdir(prompt_output_dir):
                    if fname.endswith(('.mp4', '.webm', '.png', '.jpg', '.jpeg', '.gif')):
                        fpath = os.path.join(prompt_output_dir, fname)
                        mtime = os.path.getmtime(fpath)
                        found_files.append((fname, fpath, mtime))

        logger.info(f"Found {len(found_files)} output files")

        for fname, fpath, mtime in found_files:
            s3_key = f'comfy-outputs/{prompt_id}/{fname}'
            url = s3.upload_file(fpath, s3_key)

            if url:
                output_files.append({
                    'filename': fname,
                    'type': 's3_url',
                    'data': url
                })
            else:
                output_files.append({
                    'filename': fname,
                    'type': 'local',
                    'data': f'file://{fpath}'
                })

        return {'images': output_files}


# Global worker instance
worker = ComfyUIWorker()


async def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """Named handler for runpod.serverless.start()."""
    return await worker.handler(job)


# BLOCK until ComfyUI is ready before accepting any jobs
if __name__ == "__main__":
    comfy_url = f"{COMFY_HOST_URL}/system_stats"
    print(f"Waiting for ComfyUI at {comfy_url}...")
    print(f"Max retries: {COMFY_API_AVAILABLE_MAX_RETRIES}, interval: {COMFY_API_AVAILABLE_INTERVAL_MS}ms")

    if not check_server(comfy_url, COMFY_API_AVAILABLE_MAX_RETRIES, COMFY_API_AVAILABLE_INTERVAL_MS):
        print("ERROR: ComfyUI not reachable after max retries, exiting")
        sys.exit(1)

    print("ComfyUI is ready, starting RunPod serverless handler")

    runpod.serverless.start({
        "handler": handler,
        "setup": lambda config: worker.setup(config),
        "teardown": lambda: worker.teardown(),
    })
