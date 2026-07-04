"""
High-level API to run different SVD methods on a matrix (e.g., KV-cache block).

Supported methods:
- 'full'    : torch.linalg.svd        (exact, slow, memory heavy)
- 'lowrank' : torch.svd_lowrank      (PyTorch randomized SVD baseline)
- 'cholqr'  : randomized_svd_fp16    (16-bit randomized SVD + Cholesky-QR v6)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
from torch import Tensor

from . import svd_baselines
from . import random_cholesky_v6


SVDMethod = Literal["full", "lowrank", "cholqr"]


@dataclass
class SVDConfig:
    """
    Configuration for a single SVD call.

    Fields
    ------
    method:
        Which SVD implementation to use:
        - 'full'    : exact SVD via torch.linalg.svd
        - 'lowrank' : torch.svd_lowrank randomized SVD
        - 'cholqr'  : 16-bit randomized SVD + Cholesky-QR v6
    rank:
        Target rank k for low-rank methods ('lowrank' and 'cholqr').
        Ignored for 'full'.
    n_iter:
        Number of power/subspace iterations.
        - Used by 'lowrank' (torch.svd_lowrank)
        - Used by 'cholqr' randomized_svd_fp16
    oversample:
        Oversampling dimension added on top of rank for 'cholqr'.
        A small value (e.g. 4–8) typically improves approximation quality.
    """

    method: SVDMethod = "cholqr"
    rank: Optional[int] = None
    n_iter: int = 4
    oversample: int = 4


@torch.no_grad()
def run_svd(
    tensor: Tensor,
    config: SVDConfig,
    *,
    mean_tensor: Optional[Tensor] = None,
    power_dtype: str = "bf16",
    orth: str = "chol",
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Run a chosen SVD method on the input tensor.

    Parameters
    ----------
    tensor:
        Input matrix A with shape [..., m, n].
        In the KV-cache setting this is typically a reshaped KV block.
    config:
        SVDConfig object describing which method to use and its parameters.
    mean_tensor:
        Optional mean / baseline tensor M with the same shape as `tensor`.
        For 'cholqr', M is subtracted inside the randomized SVD:
            A' = A - M
    power_dtype:
        Only used for 'cholqr'. Controls the working precision of the
        power / projection steps. Defaults to 'bf16' to match the reported v6
        path.
        Supported values: 'fp32', 'fp16', 'bf16', 'fp8', and 'fp8_e5m2'.
    orth:
        Only used for 'cholqr'. Chooses orthogonalization backend:
        - 'chol'  : Cholesky-QR (fast, memory friendly; slightly less stable)
        - 'house' : Householder QR (more stable, heavier)

    Returns
    -------
    U, S, Vh:
        Singular vectors and values such that:
            tensor ≈ U @ diag(S) @ Vh
        Shapes:
            U  : [..., m, k]
            S  : [..., k]
            Vh : [..., k, n]
        where k is rank (for low-rank methods) or min(m, n) for full SVD.
    """
    method = config.method

    if method == "full":
        return svd_baselines.full_svd(tensor)

    if method == "lowrank":
        if config.rank is None:
            raise ValueError("SVDConfig.rank must be set when method='lowrank'.")
        return svd_baselines.lowrank_svd(
            tensor_reshaped=tensor,
            rank=config.rank,
            n_iter=config.n_iter,
        )

    if method == "cholqr":
        if config.rank is None:
            raise ValueError("SVDConfig.rank must be set when method='cholqr'.")
        return random_cholesky_v6.randomized_svd_fp16(
            tensor_reshaped=tensor,
            rank=config.rank,
            n_iter=config.n_iter,
            oversample=config.oversample,
            M=mean_tensor,
            power_dtype=power_dtype,
            orth=orth,
        )

    raise ValueError(f"Unknown SVD method: {method!r}")
