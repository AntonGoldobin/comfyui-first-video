"""Force-install sageattention 2.2.0 with --no-build-isolation, see if Modal mirror has it."""
import modal

app = modal.App("probe-sage-v2")

# Force cache busting with a sentinel
image = (
    modal.Image.from_registry("sombi/comfyui:base-torch2.8.0-cu124")
    .run_commands(
        # Try the EXACT same command as our deploy
        "pip install --no-cache-dir --break-system-packages --no-build-isolation sageattention==2.2.0 2>&1 | tail -20 || echo INSTALL_FAIL",
    )
)


@app.function(image=image, gpu="H100")
def probe():
    import os
    import torch
    import shutil
    out = []

    # Did sageattention get installed?
    try:
        import sageattention
        out.append(f"sageattention version: {getattr(sageattention, '__version__', 'NO __version__')}")
        out.append(f"sageattention path: {sageattention.__file__}")
    except Exception as e:
        out.append(f"sageattention import: FAIL ({type(e).__name__}: {e})")
        return "\n".join(out)

    out.append(f"\ntorch: {torch.__version__}")
    if torch.cuda.is_available():
        out.append(f"device: {torch.cuda.get_device_name(0)}")
        out.append(f"capability: {torch.cuda.get_device_capability(0)}")

    out.append("\n=== Kernel imports (H3 patch needs these) ===")
    for kn in ["_qattn_sm90", "_qattn_sm80", "_qattn_sm89", "_qattn_sm120"]:
        try:
            m = __import__(f"sageattention.core.{kn}", fromlist=[kn])
            out.append(f"  {kn}: OK")
        except Exception as e:
            out.append(f"  {kn}: FAIL ({type(e).__name__}: {str(e)[:120]})")

    for kn in ["sageattn_qk_int8_pv_fp8_cuda", "sageattn_qk_int8_pv_fp8_cuda_pp", "sageattn_qk_int8_pv_fp16_cuda", "sageattn3"]:
        try:
            import sageattention.core as core
            v = getattr(core, kn, None)
            out.append(f"  {kn}: {'OK' if v else 'MISSING attr'}")
        except Exception as e:
            out.append(f"  {kn}: FAIL ({type(e).__name__}: {str(e)[:120]})")

    out.append("\n=== Build env ===")
    out.append(f"nvcc: {shutil.which('nvcc') or 'NOT INSTALLED'}")
    out.append(f"TORCH_CUDA_ARCH_LIST: {os.environ.get('TORCH_CUDA_ARCH_LIST', 'NOT SET')}")
    out.append(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'NOT SET')}")
    return "\n".join(out)
