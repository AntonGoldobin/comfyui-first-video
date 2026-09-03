"""Modal probe — runs inside the H3 image and reports sageattention state."""
import modal
import os
import shutil

app = modal.App("probe-sage")

image = (
    modal.Image.from_registry("sombi/comfyui:base-torch2.8.0-cu124")
    .run_commands(
        "pip install --no-cache-dir --break-system-packages sageattention || true",
    )
)


@app.function(image=image, gpu="H100")
def probe():
    import torch
    try:
        import sageattention
        sa_version = getattr(sageattention, "__version__", "NO __version__")
        sa_path = sageattention.__file__
    except Exception as e:
        return f"sageattention import FAIL: {type(e).__name__}: {e}"

    out = []
    out.append(f"sageattention version: {sa_version}")
    out.append(f"sageattention path: {sa_path}")
    out.append(f"torch: {torch.__version__}")
    out.append(f"cuda: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        out.append(f"device: {torch.cuda.get_device_name(0)}")
        out.append(f"capability: {torch.cuda.get_device_capability(0)}")

    out.append("\n=== Kernel imports ===")
    for kn in ["_qattn_sm90", "_qattn_sm80", "_qattn_sm89", "_qattn_sm120"]:
        try:
            m = __import__(f"sageattention.core.{kn}", fromlist=[kn])
            out.append(f"  {kn}: OK ({type(m).__name__})")
        except Exception as e:
            out.append(f"  {kn}: FAIL ({type(e).__name__}: {str(e)[:200]})")

    for kn in ["sageattn_qk_int8_pv_fp8_cuda", "sageattn_qk_int8_pv_fp16_cuda", "sageattn_qk_int8_pv_fp8_cuda_pp"]:
        try:
            import sageattention.core as core
            v = getattr(core, kn, None)
            out.append(f"  {kn}: {'OK' if v else 'MISSING attr'}")
        except Exception as e:
            out.append(f"  {kn}: FAIL ({type(e).__name__}: {str(e)[:200]})")

    out.append("\n=== Build env ===")
    out.append(f"nvcc: {shutil.which('nvcc')}")
    out.append(f"TORCH_CUDA_ARCH_LIST: {os.environ.get('TORCH_CUDA_ARCH_LIST', 'NOT SET')}")
    out.append(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'NOT SET')}")

    return "\n".join(out)
