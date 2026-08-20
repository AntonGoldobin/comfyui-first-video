"""Debug: run ComfyUI directly and capture first 100 lines of output."""
import modal
import os
import subprocess
import time

app = modal.App("debug-comfyui")

# Match the production image
image = (
    modal.Image.from_registry("sombi/comfyui:base-torch2.8.0-cu124")
    .run_commands(
        "apt-get update && apt-get install -y --no-install-recommends git wget ca-certificates python3 python3-venv python3-pip && rm -rf /var/lib/apt/lists/*",
        "git clone --depth=1 https://github.com/kijai/ComfyUI-KJNodes /comfyui/custom_nodes/ComfyUI-KJNodes || true",
        "git clone --depth=1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite /comfyui/custom_nodes/ComfyUI-VideoHelperSuite || true",
        "git clone --depth=1 https://github.com/rgthree/rgthree-comfy /comfyui/custom_nodes/rgthree-comfy || true",
        "git clone https://github.com/Lightricks/ComfyUI-LTXVideo /comfyui/custom_nodes/ComfyUI-LTXVideo && "
        "cd /comfyui/custom_nodes/ComfyUI-LTXVideo && git checkout 2e2ac81",
    )
    .run_commands(
        "cd /comfyui && "
        "for r in custom_nodes/*/requirements.txt; do "
        "[ -f \"$r\" ] && pip install --no-cache-dir -r \"$r\" || true; "
        "done",
        "pip install --no-cache-dir opencv-python imageio_ffmpeg fastapi httpx 'starlette>=0.36'",
    )
    .entrypoint([])
)

@app.function(image=image, gpu="A100-80GB", timeout=600, startup_timeout=600, cpu=4, memory=8192)
def debug():
    """Start ComfyUI in foreground for 60s, capture all output."""
    log = []
    log.append(f"=== env: PATH={os.environ.get('PATH', 'NONE')[:200]}")
    log.append(f"=== /venv/bin/python exists: {os.path.exists('/venv/bin/python3')}")
    log.append(f"=== /comfyui/main.py exists: {os.path.exists('/comfyui/main.py')}")
    log.append(f"=== CWD: {os.getcwd()}")

    # Try the actual command used by serve()
    proc = subprocess.Popen(
        ["/venv/bin/python3", "/comfyui/main.py", "--listen", "127.0.0.1", "--port", "8188",
         "--disable-auto-launch", "--gpu-only"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    log.append(f"=== ComfyUI PID: {proc.pid}")

    # Read 30 seconds of output
    deadline = time.time() + 30
    lines = []
    while time.time() < deadline and proc.poll() is None:
        import select
        if proc.stdout and select.select([proc.stdout], [], [], 1.0)[0]:
            line = proc.stdout.readline()
            if line:
                lines.append(line.rstrip())
        if len(lines) > 100:
            break

    # Capture remaining output if any
    try:
        rest, _ = proc.communicate(timeout=5)
        if rest:
            lines.extend(rest.splitlines()[:50])
    except subprocess.TimeoutExpired:
        proc.kill()

    log.append(f"=== Total ComfyUI output lines: {len(lines)}")
    log.append("=== First 60 lines of ComfyUI output:")
    log.extend(lines[:60])
    if proc.poll() is not None:
        log.append(f"=== ComfyUI exited with code: {proc.returncode}")
    else:
        proc.kill()
        log.append("=== ComfyUI still running (killed)")

    return "\n".join(log)

@app.local_entrypoint()
def main():
    out = debug.remote()
    print(out)
