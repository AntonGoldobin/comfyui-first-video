"""
ComfyUI RunPod Serverless Handler

Implements the RunPod Serverless Worker interface for ComfyUI.
Receives workflow JSON, submits to ComfyUI API, waits for completion,
uploads outputs to S3, returns signed URLs.

RunPod expects:
- POST /run   -> {id: string}  (submit job)
- GET /status/{id} -> {status, output?}  (poll status)

This handler implements the ServerlessWorker interface from runpod SDK.
"""

import os
import json
import asyncio
import logging
import time
import uuid
from typing import Optional

import runpod
import httpx
import boto3
from botocore.config import Config as BotoConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ComfyUI API base URL (assumes ComfyUI is running as sidecar)
COMFYUI_URL = os.environ.get('COMFYUI_URL', 'http://localhost:8188')
COMFYUI_API_KEY = os.environ.get('COMFYUI_API_KEY')

# S3 configuration from environment (fallback if not in job input)
DEFAULT_S3_BUCKET = os.environ.get('S3_BUCKET')
DEFAULT_S3_ACCESS_KEY = os.environ.get('S3_ACCESS_KEY_ID')
DEFAULT_S3_SECRET_KEY = os.environ.get('S3_SECRET_ACCESS_KEY')
DEFAULT_S3_ENDPOINT = os.environ.get('S3_ENDPOINT_URL')


class ComfyWorker:
    """
    RunPod Serverless Worker for ComfyUI.

    Receives jobs via the `handler` method, submits workflows to ComfyUI,
    polls for completion, uploads outputs to S3, and returns signed URLs.
    """

    def __init__(self):
        self.s3_client = None
        self.s3_bucket = None
        self._setup_s3_from_env()

    def _setup_s3_from_env(self):
        """Initialize S3 client from environment variables (fallback)."""
        if DEFAULT_S3_ACCESS_KEY and DEFAULT_S3_SECRET_KEY:
            try:
                self.s3_client = boto3.client(
                    's3',
                    endpoint_url=DEFAULT_S3_ENDPOINT,
                    aws_access_key_id=DEFAULT_S3_ACCESS_KEY,
                    aws_secret_access_key=DEFAULT_S3_SECRET_KEY,
                    config=BotoConfig(signature_version='s3v4')
                )
                self.s3_bucket = DEFAULT_S3_BUCKET
                logger.info(f"S3 client initialized: bucket={DEFAULT_S3_BUCKET}")
            except Exception as e:
                logger.warning(f"Failed to initialize S3 client: {e}")

    def setup(self, config):
        """
        Called once at container startup (before first job).
        Config contains the endpoint configuration from RunPod dashboard.
        """
        logger.info("ComfyWorker.setup() called")
        logger.info(f"Config keys: {list(config.keys()) if config else 'None'}")

        # Try to setup S3 from config if available
        s3_config = config.get('s3Config') if config else None
        if s3_config:
            try:
                self.s3_client = boto3.client(
                    's3',
                    endpoint_url=s3_config.get('endpointUrl'),
                    aws_access_key_id=s3_config['accessId'],
                    aws_secret_access_key=s3_config['accessSecret'],
                    config=BotoConfig(signature_version='s3v4')
                )
                self.s3_bucket = s3_config['bucketName']
                logger.info(f"S3 client initialized from config: bucket={self.s3_bucket}")
            except Exception as e:
                logger.warning(f"Failed to initialize S3 from config: {e}")

    async def handler(self, job) -> dict:
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
            s3_client = self.s3_client
            s3_bucket = self.s3_bucket
            if s3_config:
                try:
                    s3_client = boto3.client(
                        's3',
                        endpoint_url=s3_config.get('endpointUrl'),
                        aws_access_key_id=s3_config['accessId'],
                        aws_secret_access_key=s3_config['accessSecret'],
                        config=BotoConfig(signature_version='s3v4')
                    )
                    s3_bucket = s3_config['bucketName']
                except Exception as e:
                    logger.warning(f"Failed to use job S3 config: {e}, using default")
                    s3_client = self.s3_client
                    s3_bucket = self.s3_bucket

            # 1. Upload input images to /input directory if provided
            input_files = await self._upload_input_images(images)

            # 2. Submit workflow to ComfyUI
            prompt_id = await self._submit_workflow(workflow, input_files)

            # 3. Poll for completion
            result = await self._poll_for_completion(prompt_id)

            # 4. Upload outputs to S3 and return URLs
            output = await self._upload_outputs(result, prompt_id, s3_client, s3_bucket)

            logger.info(f"Job {job_id} completed successfully")
            return output

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}", exc_info=True)
            raise

    async def _upload_input_images(self, images: list) -> dict:
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

        async with httpx.AsyncClient(timeout=30) as client:
            for img in images:
                name = img.get('name', f'image_{len(input_paths)}')
                image_data = img.get('image', '')

                # If image is base64, decode and save
                if image_data:
                    import base64
                    # Handle both base64 string and URL
                    if isinstance(image_data, str) and not image_data.startswith('http'):
                        try:
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
                            resp = await client.get(image_data)
                            resp.raise_for_status()
                            img_path = f'/input/{name}'
                            with open(img_path, 'wb') as f:
                                f.write(resp.content)
                            input_paths[name] = img_path
                            logger.info(f"Downloaded input image: {name}")
                        except Exception as e:
                            logger.warning(f"Failed to download image {name}: {e}")

        return input_paths

    async def _submit_workflow(self, workflow: dict, input_files: dict) -> str:
        """
        Submit workflow to ComfyUI API.

        Args:
            workflow: The ComfyUI workflow JSON
            input_files: Dict of input file paths

        Returns:
            prompt_id from ComfyUI
        """
        # Prepare workflow - replace input image paths
        # Reelant passes images with logical names, we need to update workflow
        # to reference the uploaded files
        workflow_copy = json.loads(json.dumps(workflow))  # Deep copy

        # Update image references in LoadImage nodes
        for node_id, node_data in workflow_copy.items():
            if node_data.get('class_type') == 'LoadImage':
                inputs = node_data.get('inputs', {})
                if 'image' in inputs:
                    img_name = inputs['image']
                    if img_name in input_files:
                        inputs['image'] = input_files[img_name]

        async with httpx.AsyncClient(timeout=30) as client:
            headers = {}
            if COMFYUI_API_KEY:
                headers['Authorization'] = f'Bearer {COMFYUI_API_KEY}'

            resp = await client.post(
                f'{COMFYUI_URL}/prompt',
                json={'prompt': workflow_copy},
                headers=headers
            )
            resp.raise_for_status()

            result = resp.json()
            prompt_id = result.get('prompt_id')

            if not prompt_id:
                raise ValueError(f"No prompt_id in response: {result}")

            logger.info(f"Submitted workflow, prompt_id={prompt_id}")
            return prompt_id

    async def _poll_for_completion(self, prompt_id: str, timeout: int = 600) -> dict:
        """
        Poll ComfyUI history endpoint until workflow completes.

        Args:
            prompt_id: The prompt ID from _submit_workflow
            timeout: Maximum seconds to wait

        Returns:
            The history entry for the completed prompt
        """
        start_time = time.time()
        last_status = None

        async with httpx.AsyncClient(timeout=30) as client:
            while time.time() - start_time < timeout:
                try:
                    resp = await client.get(f'{COMFYUI_URL}/history/{prompt_id}')
                    resp.raise_for_status()
                    history = resp.json()

                    if prompt_id in history:
                        entry = history[prompt_id]
                        status = entry.get('status', {})

                        if status.get('completed'):
                            logger.info(f"Workflow {prompt_id} completed")
                            return entry

                        if status.get('executing'):
                            last_status = 'executing'
                        elif status.get('failed'):
                            error_msg = status.get('error', 'Unknown error')
                            raise RuntimeError(f"ComfyUI workflow failed: {error_msg}")

                    if time.time() - start_time > 10:  # Log every 10s after start
                        logger.info(f"Polling {prompt_id}... ({int(time.time() - start_time)}s)")

                    await asyncio.sleep(2)

                except httpx.HTTPError as e:
                    logger.warning(f"HTTP error polling history: {e}")
                    await asyncio.sleep(5)

        raise TimeoutError(f"Workflow {prompt_id} timed out after {timeout}s")

    async def _upload_outputs(
        self,
        result: dict,
        prompt_id: str,
        s3_client,
        s3_bucket: Optional[str]
    ) -> dict:
        """
        Find output files and upload to S3.

        Args:
            result: The completed workflow history entry
            prompt_id: The prompt ID
            s3_client: S3 client (or None)
            s3_bucket: S3 bucket name (or None)

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
                        # Get file modified time to find newest files (from this run)
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
            # Generate S3 URL or use local path
            if s3_client and s3_bucket:
                s3_key = f'comfy-outputs/{prompt_id}/{fname}'
                try:
                    s3_client.upload_file(
                        fpath,
                        s3_bucket,
                        s3_key,
                        ExtraArgs={'ContentType': self._get_content_type(fname)}
                    )
                    url = s3_client.generate_presigned_url(
                        'get_object',
                        Params={'Bucket': s3_bucket, 'Key': s3_key},
                        ExpiresIn=3600 * 24  # 24 hours
                    )
                    file_type = 's3_url'
                    logger.info(f"Uploaded {fname} to S3: {s3_key}")
                except Exception as e:
                    logger.warning(f"Failed to upload {fname} to S3: {e}, using local path")
                    url = f'file://{fpath}'
                    file_type = 'local'
            else:
                url = f'file://{fpath}'
                file_type = 'local'

            output_files.append({
                'filename': fname,
                'type': file_type,
                'data': url
            })

        return {'images': output_files}

    @staticmethod
    def _get_content_type(filename: str) -> str:
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


# Worker instance — imported by handler.py for runpod.serverless.start()
_worker = ComfyWorker()
