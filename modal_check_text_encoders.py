"""Verify all Gemma shards are present in volume."""
import modal
import os

app = modal.App("check-gemma")
models_volume = modal.Volume.from_name("comfyui-models", create_if_missing=True)

image = modal.Image.from_registry("sombi/comfyui:base-torch2.8.0-cu124").entrypoint([])

@app.function(image=image, volumes={"/modal-data": models_volume}, cpu=2, memory=4096, timeout=60)
def check():
    out = {}
    if os.path.isdir("/modal-data/models/text_encoders/ltx-2-gemma"):
        files = sorted(os.listdir("/modal-data/models/text_encoders/ltx-2-gemma"))
        out["gemma_files"] = []
        for f in files:
            p = f"/modal-data/models/text_encoders/ltx-2-gemma/{f}"
            out["gemma_files"].append(f"{f}: {os.path.getsize(p)/1e9:.2f}GB")
    else:
        out["text_enc_dir"] = "MISSING"
    if os.path.isdir("/modal-data/models/checkpoints"):
        out["checkpoints"] = sorted(os.listdir("/modal-data/models/checkpoints"))
    return out

@app.local_entrypoint()
def main():
    r = check.remote()
    for k, v in r.items():
        print(f"=== {k} ===")
        if isinstance(v, list):
            for x in v:
                print(f"  {x}")
        else:
            print(f"  {v}")
