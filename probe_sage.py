import os
import shutil
import sageattention
print("=== sageattention ===")
print("version:", getattr(sageattention, "__version__", "NO __version__"))
print("module path:", sageattention.__file__)
print("dir (no _):", [x for x in dir(sageattention) if not x.startswith("_")])

import torch
print("\n=== torch / CUDA ===")
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))

print("\n=== Kernel imports ===")
for kn in ["_qattn_sm90", "_qattn_sm80", "_qattn_sm89", "_qattn_sm120"]:
    try:
        m = __import__(f"sageattention.core.{kn}", fromlist=[kn])
        print(f"  {kn}: OK -> {type(m)}")
    except Exception as e:
        print(f"  {kn}: FAIL -> {type(e).__name__}: {e}")

for kn in ["sageattn_qk_int8_pv_fp8_cuda", "sageattn_qk_int8_pv_fp16_cuda", "sageattn_qk_int8_pv_fp8_cuda_pp"]:
    try:
        from sageattention.core import __getattr__ as _
        m = __import__("sageattention.core", fromlist=[kn])
        v = getattr(m, kn, None)
        print(f"  {kn}: {'OK' if v else 'NONE (attr missing)'}")
    except Exception as e:
        print(f"  {kn}: FAIL -> {type(e).__name__}: {e}")

print("\n=== Build env ===")
print("nvcc:", shutil.which("nvcc"))
print("TORCH_CUDA_ARCH_LIST:", os.environ.get("TORCH_CUDA_ARCH_LIST", "NOT SET"))
print("CUDA_HOME:", os.environ.get("CUDA_HOME", "NOT SET"))
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "NOT SET"))
