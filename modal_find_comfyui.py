"""Find where ComfyUI actually lives in the sombi base."""
import modal
import os
import subprocess

app = modal.App("find-comfyui")

image = modal.Image.from_registry("sombi/comfyui:base-torch2.8.0-cu124")

@app.function(image=image, timeout=120)
def find():
    out = {}
    # Show /comfyui contents if it exists
    if os.path.isdir("/comfyui"):
        out["comfyui_exists"] = True
        out["comfyui_contents"] = sorted(os.listdir("/comfyui"))[:30]
    else:
        out["comfyui_exists"] = False
    # Show /workspace contents
    if os.path.isdir("/workspace"):
        out["workspace_contents"] = sorted(os.listdir("/workspace"))[:30]
    # Check for ComfyUI dirs anywhere
    r = subprocess.run(["find", "/", "-maxdepth", "3", "-name", "main.py", "-path", "*ComfyUI*"],
                       capture_output=True, text=True, timeout=30)
    out["comfyui_main_files"] = r.stdout.splitlines()[:10]
    # Check start.sh
    if os.path.isfile("/start.sh"):
        with open("/start.sh") as f:
            out["start_sh"] = f.read()[:2000]
    return out

@app.local_entrypoint()
def main():
    r = find.remote()
    for k, v in r.items():
        print(f"=== {k} ===")
        if isinstance(v, list):
            for x in v:
                print(f"  {x}")
        else:
            print(f"  {v}")
