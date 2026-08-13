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

# IMPORTANT: bridge env vars BEFORE `import runpod`. The runpod SDK reads
# RUNPOD_POD_ID, RUNPOD_AI_API_KEY, and RUNPOD_WEBHOOK_GET_JOB at MODULE
# IMPORT time (worker_state.WORKER_ID and rp_job.JOB_GET_URL are computed
# once when those modules are first imported). If we set them after the
# import, the constants are already wrong (random UUID instead of endpoint
# id, JOB_GET_URL stuck at "$ID" placeholder).
import os
if not os.environ.get("RUNPOD_POD_ID") and os.environ.get("RUNPOD_ENDPOINT_ID"):
    os.environ["RUNPOD_POD_ID"] = os.environ["RUNPOD_ENDPOINT_ID"]
if not os.environ.get("RUNPOD_AI_API_KEY"):
    for alt in ("RUNPOD_FLASH_API_KEY", "RUNPOD_SERVERLESS_API_KEY"):
        if os.environ.get(alt):
            os.environ["RUNPOD_AI_API_KEY"] = os.environ[alt]
            break
if not os.environ.get("RUNPOD_WEBHOOK_GET_JOB") and os.environ.get("RUNPOD_ENDPOINT_ID"):
    eid = os.environ["RUNPOD_ENDPOINT_ID"]
    # SDK uses .replace("$ID", WORKER_ID) — must be literal $ID, not $RUNPOD_POD_ID.
    os.environ["RUNPOD_WEBHOOK_GET_JOB"] = (
        f"https://api.runpod.ai/v2/{eid}/job-take/$ID"
    )

import asyncio
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

# RunPod env bridging is at the top of this file (before `import runpod`) so
# the SDK captures the values at module-import time. Don't add it here.


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

    # Verbose diagnostic: log first HTTP error in detail (Aug 10 2026 — Phase 15 close-out).
    # Without this the runpod serverless log shows only "Waiting for ComfyUI... (300/300)"
    # with no indication of whether the failure was ConnectionRefused, Timeout, or HTTP 5xx.
    _logged_first_error = False

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
            # Non-200 — log it on the first occurrence only to avoid log spam.
            if not _logged_first_error:
                logger.warning(
                    f"[DIAG] ComfyUI returned non-200 on attempt {attempt}: "
                    f"status={response.status_code} body={response.text[:200]!r}"
                )
                _logged_first_error = True
        except requests.RequestException as exc:
            if not _logged_first_error:
                logger.warning(
                    f"[DIAG] ComfyUI HTTP error on attempt {attempt}: "
                    f"{type(exc).__name__}: {exc}"
                )
                _logged_first_error = True

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
        # Initialize httpx client immediately (setup() may not be called before first job)
        self.httpx_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )

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

        # Fast-fail validation (Phase 19): recognize synthetic probes and
        # short-circuit before spinning up ComfyUI. Without this, every
        # test probe runs the full ComfyUI startup (5–30 s of GPU-seconds
        # wasted per probe) and either:
        #   - returns ValueError("No workflow found") from below
        #   - or worse, runs a partial workflow and fails mid-execution
        # Recognized probe keys (any one of):
        #   - flash_smoke_test: bool   — RunPod deploy smoke test (CI)
        #   - dry_run: bool            — explicit no-op from Reelant
        #   - ping / healthcheck: bool — kubernetes-style liveness probe
        if any(job_input.get(k) for k in ("flash_smoke_test", "dry_run", "ping", "healthcheck")):
            logger.info(f"Job {job_id}: synthetic probe detected — short-circuiting")
            return {
                "status": "ok",
                "probe": True,
                "job_id": job_id,
                "msg": "smoke test fast-path (Phase 19)",
            }

        # Phase 26: list_nodes probe — runs ComfyUI /object_info so the
        # caller can see which custom node classes are actually registered
        # (no point waiting 12 min for an LTX job only to discover the
        # node pack didn't install). Returns the count + a sample of class
        # names so we can grep for LTXV* / VHS_*. Requires ComfyUI ready.
        if job_input.get("list_nodes"):
            logger.info(f"Job {job_id}: list_nodes probe — querying /object_info")
            resp = await self.httpx_client.get(
                f'{COMFY_HOST_URL}/object_info',
                headers=get_api_headers(),
            )
            resp.raise_for_status()
            info = resp.json()
            classes = sorted(info.keys())
            ltxv = [c for c in classes if c.startswith("LTXV")]
            vhs = [c for c in classes if c.startswith("VHS_") or "VideoHelper" in c]
            return {
                "status": "ok",
                "probe": "list_nodes",
                "job_id": job_id,
                "total_classes": len(classes),
                "ltxv_count": len(ltxv),
                "ltxv_classes": ltxv,
                "vhs_count": len(vhs),
                "vhs_classes": vhs,
                "sample": classes[:20],
            }

        # Phase 28: bootstrap_models probe — runs download-models-ltx2.sh on
        # demand. Use when the entrypoint-wrapper.sh cold-start bootstrap
        # silently failed (or for fresh network volumes that haven't seen
        # a cold start). Returns ls of model dirs + download log tail.
        if job_input.get("bootstrap_models"):
            logger.info(f"Job {job_id}: bootstrap_models — running download script")
            import asyncio

            async def run_bootstrap():
                proc = await asyncio.create_subprocess_exec(
                    "/usr/local/bin/download-models-ltx2.sh",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                stdout, _ = await proc.communicate()
                return proc.returncode, stdout.decode("utf-8", errors="replace")

            try:
                rc, log = await run_bootstrap()
            except Exception as e:
                return {"status": "error", "probe": "bootstrap_models",
                        "error": str(e), "job_id": job_id}

            # After download, ls the model dirs
            import os
            def safe_ls(p):
                if not os.path.isdir(p):
                    return f"missing: {p}"
                try:
                    entries = os.listdir(p)
                    return sorted(entries)
                except Exception as e:
                    return f"err: {e}"

            return {
                "status": "ok" if rc == 0 else "download-failed",
                "probe": "bootstrap_models",
                "job_id": job_id,
                "returncode": rc,
                "log_tail": log[-3000:] if log else "",
                "models": {
                    "checkpoints": safe_ls("/runpod-volume/models/checkpoints"),
                    "text_encoders": safe_ls("/runpod-volume/models/text_encoders"),
                    "latent_upscale_models": safe_ls("/runpod-volume/models/latent_upscale_models"),
                    "vae": safe_ls("/runpod-volume/models/vae"),
                    "diffusion_models": safe_ls("/runpod-volume/models/diffusion_models"),
                },
                "disk_usage_gb": round(
                    sum(
                        os.path.getsize(os.path.join(dp, f))
                        for dp, _, fns in os.walk("/runpod-volume/models")
                        for f in fns
                    ) / 1e9, 2
                ) if os.path.isdir("/runpod-volume/models") else None,
            }

        # Phase 28i: bash_command probe — runs arbitrary shell command for
        # diagnostics. Use sparingly. ComfyUI must be ready (not required).
        if job_input.get("bash_command"):
            cmd = job_input["bash_command"]
            if not isinstance(cmd, str) or not cmd.strip():
                return {"status": "error", "probe": "bash_command",
                        "error": "bash_command must be a non-empty string"}
            import asyncio
            try:
                proc = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
                return {
                    "status": "ok",
                    "probe": "bash_command",
                    "job_id": job_id,
                    "returncode": proc.returncode,
                    "stdout": stdout.decode("utf-8", errors="replace"),
                }
            except asyncio.TimeoutError:
                return {"status": "error", "probe": "bash_command",
                        "error": "command timed out (60s)", "job_id": job_id}
            except Exception as e:
                return {"status": "error", "probe": "bash_command",
                        "error": str(e), "job_id": job_id}

        # Fast-fail for missing workflow (saves 5-30s of ComfyUI startup
        # for malformed probes that don't set any probe flag).
        if not (job_input.get("workflow") or job_input.get("prompt")):
            recognized = {"flash_smoke_test", "dry_run", "ping", "healthcheck",
                          "list_nodes", "bootstrap_models", "bash_command",
                          "workflow", "prompt", "images", "s3Config"}
            raise ValueError(
                f"No workflow found in job input. "
                f"Recognized keys: {sorted(recognized)}. "
                f"Got: {sorted(job_input.keys())}"
            )

        try:
            # Extract workflow and config from job input
            workflow = job_input.get('workflow') or job_input.get('prompt')
            images = job_input.get('images', [])
            s3_config = job_input.get('s3Config')

            # Handle double-serialization: if workflow is a string, parse it
            if isinstance(workflow, str):
                logger.warning(f"Job {job_id}: workflow received as string, parsing...")
                try:
                    workflow = json.loads(workflow)
                except json.JSONDecodeError as e:
                    raise ValueError(f"workflow is a string but not valid JSON: {e}")

            if not workflow:
                raise ValueError("No workflow found in job input")

            if not isinstance(workflow, dict):
                raise ValueError(f"workflow is not a dict after parsing: {type(workflow)}")

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

        # Handle new ComfyUI workflow format (nodes as list)
        # Convert from: {"nodes": [{"id": 2, "type": "LoadImage", ...}, ...]}
        # To old format: {"2": {"class_type": "LoadImage", "inputs": {...}}, ...}
        if isinstance(workflow_copy.get('nodes'), list):
            logger.info("Converting workflow from new format to old format")
            old_format = {}
            for node in workflow_copy['nodes']:
                node_id = str(node.get('id', ''))
                old_format[node_id] = {
                    'class_type': node.get('type', ''),
                    'inputs': node.get('inputs', {}),
                }
            workflow_copy = old_format
            logger.info(f"Converted {len(workflow_copy)} nodes to old format")

        # Update image references in LoadImage nodes
        for node_id, node_data in workflow_copy.items():
            if node_data.get('class_type') == 'LoadImage':
                inputs = node_data.get('inputs', {})
                if 'image' in inputs:
                    img_name = inputs['image']
                    if img_name in input_files:
                        inputs['image'] = input_files[img_name]
                        logger.info(f"Replaced image path for {img_name}: {input_files[img_name]}")

        logger.info(f"===== DEBUG: Submitting workflow to {COMFY_HOST_URL}/prompt =====")
        logger.info(f"Workflow payload: {json.dumps(workflow_copy, indent=2)}")
        logger.info(f"===== END DEBUG =====")

        resp = await self.httpx_client.post(
            f'{COMFY_HOST_URL}/prompt',
            json={'prompt': workflow_copy},
            headers=get_api_headers()
        )

        logger.info(f"Response status: {resp.status_code}")
        if resp.status_code != 200:
            body = resp.text
            logger.error(f"Response body: {body}")
            # Re-raise with body in the message so RunPod's error_message captures it
            # (otherwise the actual ComfyUI rejection reason is lost — we only see
            # "400 Bad Request" without the validation detail).
            raise httpx.HTTPStatusError(
                f"ComfyUI /prompt returned {resp.status_code} {resp.reason_phrase} — body: {body[:1500]}",
                request=resp.request,
                response=resp,
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

        # Find outputs in /comfyui/output (set via --base-directory and --output-directory)
        # Also check legacy locations for backwards compatibility
        output_dirs = ['/comfyui/output', '/output', '/workspace/ComfyUI/output', os.path.expanduser('~/ComfyUI/output')]

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
