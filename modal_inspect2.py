"""Find the actual Python venv path in sombi base."""
import modal
import os
import subprocess

app = modal.App("inspect2")

image = modal.Image.from_registry("sombi/comfyui:base-torch2.8.0-cu124")

@app.function(image=image, timeout=120)
def inspect():
    out = {}
    # Search common venv locations
    for venv in ["/venv", "/opt/venv", "/comfyui/.venv", "/comfyui/venv", "/workspace/venv"]:
        if os.path.isdir(venv):
            py = f"{venv}/bin/python"
            out[venv] = {
                "exists": os.path.isdir(venv),
                "python": os.path.exists(py),
                "python_executable": os.access(py, os.X_OK) if os.path.exists(py) else False,
            }
    # Show root dirs
    out["root"] = sorted(os.listdir("/"))
    # What does `which python3` say?
    r = subprocess.run(["which", "python3"], capture_output=True, text=True, timeout=10)
    out["which_python3"] = r.stdout.strip()
    # Try running python3 --version
    r = subprocess.run(["python3", "--version"], capture_output=True, text=True, timeout=10)
    out["python3_version"] = (r.stdout + r.stderr).strip()
    # Look for /comfyui structure
    if os.path.isdir("/comfyui"):
        out["comfyui_top"] = sorted(os.listdir("/comfyui"))[:20]
        if os.path.isfile("/comfyui/main.py"):
            out["comfyui_main_size"] = os.path.getsize("/comfyui/main.py")
    # Check /venv specifically
    if os.path.isdir("/venv"):
        out["venv_bin"] = sorted(os.listdir("/venv/bin"))[:20]
    return out

@app.local_entrypoint()
def main():
    r = inspect.remote()
    for k, v in r.items():
        print(f"=== {k} ===")
        if isinstance(v, dict):
            for kk, vv in v.items():
                print(f"  {kk}: {vv}")
        elif isinstance(v, list):
            for x in v:
                print(f"  {x}")
        else:
            print(f"  {v}")
