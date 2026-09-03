"""Probe: install SageAttention 2.2.0 from git on H100 + verify sm90 FP8 kernel.

Strategy:
1. Install nvcc + ninja + cuda dev tools (SageAttention needs to compile CUDA kernels from source)
2. git+pip install main branch of thu-ml/SageAttention (2.2.0 is on main, not on PyPI)
3. Verify _qattn_sm90 and sageattn_qk_int8_pv_fp8_cuda_sm90 modules are importable
"""
import modal

app = modal.App("probe-sage-v5")

image = (
    modal.Image.from_registry("sombi/comfyui:base-torch2.8.0-cu124")
    .run_commands(
        # Install CUDA dev tools + git
        "apt-get update -qq && apt-get install -y -qq --no-install-recommends "
        "git ninja-build 2>&1 | tail -3",
        # CRITICAL: set TORCH_CUDA_ARCH_LIST BEFORE pip install.
        # Modal build workers have NO GPU, so torch.utils.cpp_extension can't
        # auto-detect compute capability → build dies with "No target compute capabilities".
        # H100 = Hopper = sm_90. We also include 8.0 (A100) and 8.9 (L40) for portability.
        'export TORCH_CUDA_ARCH_LIST="9.0+PTX"',
        # Install SageAttention from git main (gives 2.2.0 with SM90 FP8)
        # --no-build-isolation is critical (otherwise build uses wrong torch headers)
        "TORCH_CUDA_ARCH_LIST='9.0+PTX' pip install --no-cache-dir --break-system-packages "
        "--no-build-isolation git+https://github.com/thu-ml/SageAttention.git "
        "2>&1 | tail -25 || echo INSTALL_FAILED",
    )
)


@app.function(image=image, gpu="H100", timeout=1800)  # 30min — compile can be slow
def probe():
    import os
    import torch
    import shutil

    print("=" * 60, flush=True)

    try:
        import sageattention
        print(f"sageattention version: {getattr(sageattention, '__version__', 'NO __version__')}", flush=True)
        print(f"path: {sageattention.__file__}", flush=True)
    except Exception as e:
        print(f"sageattention import FAIL: {type(e).__name__}: {e}", flush=True)
        return "no_sage"

    if torch.cuda.is_available():
        print(f"torch: {torch.__version__}", flush=True)
        print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"capability: {torch.cuda.get_device_capability(0)}", flush=True)

    # CUDA dev tools (needed for source build verification)
    print(f"\nnvcc: {shutil.which('nvcc') or 'NOT INSTALLED'}", flush=True)
    print(f"ninja: {shutil.which('ninja') or 'NOT INSTALLED'}", flush=True)
    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'NOT SET')}", flush=True)
    print(f"TORCH_CUDA_ARCH_LIST: {os.environ.get('TORCH_CUDA_ARCH_LIST', 'NOT SET')}", flush=True)

    print("\n=== SM-specific kernels (H100 = sm90) ===", flush=True)
    for kn in ["_qattn_sm90", "_qattn_sm80", "_qattn_sm89", "_qattn_sm120"]:
        try:
            m = __import__(f"sageattention.core.{kn}", fromlist=[kn])
            print(f"  {kn}: OK", flush=True)
        except Exception as e:
            print(f"  {kn}: FAIL ({type(e).__name__}: {str(e)[:200]})", flush=True)

    print("\n=== FP8 kernels (H3 patch uses these) ===", flush=True)
    for kn in [
        "sageattn_qk_int8_pv_fp8_cuda",
        "sageattn_qk_int8_pv_fp8_cuda_sm90",   # H100-optimized
        "sageattn_qk_int8_pv_fp8_cuda_pp",
        "sageattn_qk_int8_pv_fp16_cuda",
        "sageattn3",                              # SA3 (Blackwell-only, will likely FAIL on H100)
    ]:
        try:
            import sageattention.core as core
            v = getattr(core, kn, None)
            print(f"  {kn}: {'OK' if v else 'MISSING attr'}", flush=True)
        except Exception as e:
            print(f"  {kn}: FAIL ({type(e).__name__}: {str(e)[:200]})", flush=True)

    # Look for KJNodes H3 patch file and check what version it actually requires
    print("\n=== KJNodes H3 patch inspection ===", flush=True)
    try:
        import ComfyUI_KJNodes
        kj_path = os.path.dirname(ComfyUI_KJNodes.__file__)
        print(f"KJNodes path: {kj_path}", flush=True)
        import subprocess
        result = subprocess.run(
            ["grep", "-rn", "sageattention is not new enough", kj_path],
            capture_output=True, text=True, timeout=10
        )
        if result.stdout.strip():
            for line in result.stdout.strip().split("\n")[:5]:
                print(f"  {line}", flush=True)
        else:
            print("  (search string not found in KJNodes)", flush=True)
        # Find H3 patch file
        result = subprocess.run(
            ["find", kj_path, "-name", "*ageAttention*", "-o", "-name", "*H3*"],
            capture_output=True, text=True, timeout=10
        )
        print(f"\n  H3/sage files in KJNodes:", flush=True)
        for line in result.stdout.strip().split("\n")[:10]:
            print(f"    {line}", flush=True)
    except Exception as e:
        print(f"KJNodes check: {e}", flush=True)

    return "done"
