import torch
import os

# With TORCH_LIBRARY, we need to explicitly load the library using torch.ops.load_library
# Find the .so file
try:
    import ops
    ops_dir = os.path.dirname(ops.__file__)
    # Find the .so file (name varies by Python version)
    so_files = [f for f in os.listdir(ops_dir) if f.startswith('_shadowkv') and f.endswith('.so')]
    if so_files:
        so_path = os.path.join(ops_dir, so_files[0])
        torch.ops.load_library(so_path)
    else:
        import warnings
        warnings.warn("Could not find _shadowkv.so extension file")
except Exception as e:
    # This can happen during installation or if the extension isn't built yet
    import warnings
    warnings.warn(f"Could not load _shadowkv extension: {e}")

from .tensor_op import (
    batch_gather_gemm_cuda,
    batch_gather_gemm_rotary_pos_emb_cuda,
    apply_rotary_pos_emb_single,
    repeat_kv,
)

__all__ = [
    'batch_gather_gemm_cuda',
    'batch_gather_gemm_rotary_pos_emb_cuda',
    'apply_rotary_pos_emb_single',
    'repeat_kv',
]

