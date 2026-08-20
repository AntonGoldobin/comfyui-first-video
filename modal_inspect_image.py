"""Override entrypoint to inspect image."""
import modal
import os
import subprocess

app = modal.App("image-inspect")

# Override entrypoint to bash sleep loop (Modal appends user flags to tail, so use bash)
image = (
    modal.Image.from_registry("antongoldobin/comfyui-ltx-video:latest")
    .entrypoint(["/bin/bash", "-c", "while true; do sleep 60; done"])
)

@app.function(image=image, startup_timeout=600, timeout=600)
def inspect():
    out = {}
    # Find all python binaries
    r = subprocess.run(["find", "/", "-name", "python*", "-type", "f", "-executable"],
                       capture_output=True, text=True, timeout=30)
    out["python_files"] = r.stdout.splitlines()[:20]
    # Find comfyui
    r = subprocess.run(["find", "/", "-name", "main.py", "-path", "*comfy*"],
                       capture_output=True, text=True, timeout=30)
    out["comfyui_main"] = r.stdout.splitlines()[:5]
    # Top-level dirs
    out["root_dirs"] = sorted(os.listdir("/"))
    return out

@app.local_entrypoint()
def main():
    r = inspect.remote()
    print("=== Python files ===")
    for p in r["python_files"]:
        print(f"  {p}")
    print("=== ComfyUI main.py ===")
    for p in r["comfyui_main"]:
        print(f"  {p}")
    print("=== Root dirs ===")
    print(f"  {r['root_dirs']}")