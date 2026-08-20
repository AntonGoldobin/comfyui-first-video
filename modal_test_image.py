"""Quick test: check what Python/ComfyUI paths exist in the image."""
import modal
import os

app = modal.App("image-test")

image = modal.Image.from_registry("antongoldobin/comfyui-ltx-video:latest")

@app.function(image=image, timeout=120)
def inspect():
    paths_to_check = [
        "/opt/venv/bin/python",
        "/workspace/venv/bin/python",
        "/comfyui/venv/bin/python",
        "/usr/local/bin/python3",
        "/usr/bin/python3",
        "/usr/local/bin/python",
        "/usr/bin/python",
        "/comfyui/main.py",
    ]
    result = {}
    for p in paths_to_check:
        if os.path.exists(p):
            sz = os.path.getsize(p) if os.path.isfile(p) else "DIR"
            result[p] = sz
        else:
            result[p] = "MISSING"
    return result

@app.local_entrypoint()
def main():
    r = inspect.remote()
    print("Image contents:")
    for k, v in r.items():
        print(f"  {k}: {v}")