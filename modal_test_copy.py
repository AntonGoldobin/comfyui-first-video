"""Test if ComfyUI sees a copied (not symlinked) checkpoint."""
import modal
import os

app = modal.App("test-copy")

image = (
    modal.Image.from_registry("sombi/comfyui:base-torch2.8.0-cu124")
    .run_commands("apt-get update && apt-get install -y python3 2>&1 | tail -1")
    .run_commands("pip install --no-cache-dir httpx fastapi 'starlette>=0.36'")
    .entrypoint([])
)
models_volume = modal.Volume.from_name("comfyui-models", create_if_missing=True)

@app.function(image=image, gpu="A100-80GB", volumes={"/modal-data": models_volume},
              cpu=4, memory=8192, timeout=600, startup_timeout=600)
def test():
    import subprocess
    # Copy (not symlink) a small test checkpoint
    src = "/modal-data/models/checkpoints/ltx-2-19b-distilled.safetensors"
    dst = "/ComfyUI/models/checkpoints/ltx-2-19b-distilled.safetensors"
    if not os.path.exists(dst):
        # Try to hardlink (instant, shares inode) — works if same FS
        try:
            os.link(src, dst)
            kind = "hardlink"
        except OSError:
            # Fall back to copy (slow but works across filesystems)
            print(f"link failed, copying... this will take ~1 min")
            import shutil
            shutil.copy2(src, dst)
            kind = "copy"
        print(f"Created {kind}: {dst}")
    else:
        print(f"Already exists: {dst}")
    # Start ComfyUI
    proc = subprocess.Popen(
        ["/venv/bin/python3", "/ComfyUI/main.py", "--listen", "127.0.0.1", "--port", "8188",
         "--disable-auto-launch", "--gpu-only"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    # Wait for ready
    import time
    import httpx
    for i in range(600):
        try:
            r = httpx.get("http://localhost:8188/system_stats", timeout=2)
            if r.status_code == 200:
                print(f"ComfyUI ready after {i+1}s")
                break
        except Exception:
            pass
        time.sleep(1)
    # Check checkpoints
    r = httpx.get("http://localhost:8188/models/checkpoints", timeout=10)
    print(f"/models/checkpoints response: {r.status_code} {r.text[:500]}")
    proc.kill()
    return {"kind": kind, "checkpoints": r.text}

@app.local_entrypoint()
def main():
    r = test.remote()
    print(r)
