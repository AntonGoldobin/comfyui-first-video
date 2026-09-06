"""Try installing sageattention 2.2.0 from pypi.org directly (bypassing Modal mirror)."""
import modal

app = modal.App("probe-sage-v3")

image = (
    modal.Image.from_registry("sombi/comfyui:base-torch2.8.0-cu124")
    .run_commands(
        # Force pypi.org, see if external PyPI is reachable
        "pip install --no-cache-dir --break-system-packages --no-build-isolation "
        "--index-url https://pypi.org/simple/ "
        "sageattention==2.2.0 2>&1 | tail -10 || echo INSTALL_FAIL",
    )
)


@app.function(image=image, gpu="H100")
def probe():
    import torch
    import os
    import shutil
    out = []

    try:
        import sageattention
        out.append(f"sageattention version: {getattr(sageattention, '__version__', 'NO __version__')}")
        out.append(f"path: {sageattention.__file__}")
    except Exception as e:
        out.append(f"sageattention import FAIL: {e}")
        return "\n".join(out)

    if torch.cuda.is_available():
        out.append(f"torch: {torch.__version__}")
        out.append(f"device: {torch.cuda.get_device_name(0)}")
        out.append(f"capability: {torch.cuda.get_device_capability(0)}")

    out.append("\n=== Kernel imports ===")
    for kn in ["_qattn_sm90", "_qattn_sm80", "_qattn_sm120"]:
        try:
            m = __import__(f"sageattention.core.{kn}", fromlist=[kn])
            out.append(f"  {kn}: OK")
        except Exception as e:
            out.append(f"  {kn}: FAIL ({type(e).__name__}: {str(e)[:150]})")

    for kn in ["sageattn_qk_int8_pv_fp8_cuda", "sageattn_qk_int8_pv_fp8_cuda_pp", "sageattn3"]:
        try:
            import sageattention.core as core
            v = getattr(core, kn, None)
            out.append(f"  {kn}: {'OK' if v else 'MISSING attr'}")
        except Exception as e:
            out.append(f"  {kn}: FAIL ({type(e).__name__}: {str(e)[:150]})")

    out.append(f"\nnvcc: {shutil.which('nvcc') or 'NOT INSTALLED'}")
    out.append(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'NOT SET')}")
    return "\n".join(out)
