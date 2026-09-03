"""Print all probe info to stdout (modal run only shows return value, but stdout shows in logs)."""
import modal
import sys

app = modal.App("probe-sage-v4")

image = (
    modal.Image.from_registry("sombi/comfyui:base-torch2.8.0-cu124")
    .run_commands(
        # Bypass Modal mirror, install from pypi.org directly
        "pip install --no-cache-dir --break-system-packages --no-build-isolation "
        "--index-url https://pypi.org/simple/ "
        "sageattention==2.2.0 2>&1 | tee /tmp/sage_install.log || echo INSTALL_FAILED",
    )
)


@app.function(image=image, gpu="H100")
def probe():
    import torch
    import os
    import shutil

    print("=" * 60, flush=True)

    # Show install log if it exists
    try:
        with open("/tmp/sage_install.log") as f:
            log = f.read()
        print("=== INSTALL LOG (last 30 lines) ===", flush=True)
        for line in log.splitlines()[-30:]:
            print(line, flush=True)
    except Exception as e:
        print(f"no install log: {e}", flush=True)

    try:
        import sageattention
        print(f"\nsageattention version: {sageattention.__version__}", flush=True)
        print(f"path: {sageattention.__file__}", flush=True)
    except Exception as e:
        print(f"sageattention import FAIL: {e}", flush=True)
        return "fail"

    if torch.cuda.is_available():
        print(f"torch: {torch.__version__}", flush=True)
        print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"capability: {torch.cuda.get_device_capability(0)}", flush=True)

    print("\n=== Kernel imports (H3 patch needs these) ===", flush=True)
    for kn in ["_qattn_sm90", "_qattn_sm80", "_qattn_sm89", "_qattn_sm120"]:
        try:
            m = __import__(f"sageattention.core.{kn}", fromlist=[kn])
            print(f"  {kn}: OK", flush=True)
        except Exception as e:
            print(f"  {kn}: FAIL ({type(e).__name__}: {str(e)[:150]})", flush=True)

    for kn in ["sageattn_qk_int8_pv_fp8_cuda", "sageattn_qk_int8_pv_fp8_cuda_pp", "sageattn_qk_int8_pv_fp16_cuda", "sageattn3"]:
        try:
            import sageattention.core as core
            v = getattr(core, kn, None)
            print(f"  {kn}: {'OK' if v else 'MISSING attr'}", flush=True)
        except Exception as e:
            print(f"  {kn}: FAIL ({type(e).__name__}: {str(e)[:150]})", flush=True)

    print(f"\nnvcc: {shutil.which('nvcc') or 'NOT INSTALLED'}", flush=True)
    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'NOT SET')}", flush=True)
    return "done"
