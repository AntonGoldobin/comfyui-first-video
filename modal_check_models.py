"""Verify symlink + checkpoint visibility."""
import modal
import os

app = modal.App("check-models")

image = (
    modal.Image.from_registry("sombi/comfyui:base-torch2.8.0-cu124")
    .run_commands("apt-get update && apt-get install -y python3 2>&1 | tail -1")
    .run_commands(
        "git clone --depth=1 https://github.com/kijai/ComfyUI-KJNodes /ComfyUI/custom_nodes/ComfyUI-KJNodes || true",
        "git clone --depth=1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite /ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite || true",
        "git clone --depth=1 https://github.com/rgthree/rgthree-comfy /ComfyUI/custom_nodes/rgthree-comfy || true",
        "git clone https://github.com/Lightricks/ComfyUI-LTXVideo /ComfyUI/custom_nodes/ComfyUI-LTXVideo && "
        "cd /ComfyUI/custom_nodes/ComfyUI-LTXVideo && git checkout 2e2ac81",
    )
    .run_commands(
        "cd /ComfyUI && for r in custom_nodes/*/requirements.txt; do "
        "[ -f \"$r\" ] && pip install --no-cache-dir -r \"$r\" || true; done",
        "pip install --no-cache-dir opencv-python imageio_ffmpeg fastapi httpx 'starlette>=0.36'",
    )
    .entrypoint([])
)

models_volume = modal.Volume.from_name("comfyui-models", create_if_missing=True)

@app.function(image=image, gpu="A100-80GB", volumes={"/modal-data": models_volume},
              cpu=4, memory=8192, timeout=300)
def check():
    out = {}
    # Check volume contents
    if os.path.isdir("/modal-data/models/checkpoints"):
        out["volume_checkpoints"] = sorted(os.listdir("/modal-data/models/checkpoints"))
    else:
        out["volume_dir"] = "MISSING"
    # Check image /ComfyUI/models
    out["comfyui_models_exists"] = os.path.isdir("/ComfyUI/models")
    out["comfyui_models_islink"] = os.path.islink("/ComfyUI/models")
    if os.path.isdir("/ComfyUI/models"):
        out["comfyui_models_contents"] = sorted(os.listdir("/ComfyUI/models"))
        if os.path.isdir("/ComfyUI/models/checkpoints"):
            out["ckpt_dir_contents"] = sorted(os.listdir("/ComfyUI/models/checkpoints"))
            for f in os.listdir("/ComfyUI/models/checkpoints"):
                p = f"/ComfyUI/models/checkpoints/{f}"
                out[f"ckpt_{f}_islink"] = os.path.islink(p)
                if os.path.islink(p):
                    out[f"ckpt_{f}_target"] = os.readlink(p)
                    out[f"ckpt_{f}_target_exists"] = os.path.exists(os.readlink(p))
    # Try to do the symlink ourselves
    src = "/modal-data/models/checkpoints/ltx-2-19b-distilled.safetensors"
    dst = "/ComfyUI/models/checkpoints/ltx-2-19b-distilled.safetensors"
    if os.path.exists(src) and not os.path.exists(dst):
        os.symlink(src, dst)
        out["symlink_created"] = True
        out["new_dst_exists"] = os.path.exists(dst)
    elif os.path.exists(src):
        out["symlink_already"] = True
    return out

@app.local_entrypoint()
def main():
    r = check.remote()
    for k, v in r.items():
        print(f"  {k}: {v}")
