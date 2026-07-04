from typing import Tuple

import torch
from torch import Tensor


@torch.no_grad()
def full_svd(tensor_reshaped: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Full SVD baseline using torch.linalg.svd.

    Computes the exact singular value decomposition:
        A = U @ diag(S) @ Vh

    Parameters
    ----------
    tensor_reshaped:
        Input matrix A with shape [..., m, n].

    Returns
    -------
    U, S, Vh:
        U  : [..., m, k]
        S  : [..., k]
        Vh : [..., k, n]
        where k = min(m, n).
    """
    U, S, Vh = torch.linalg.svd(tensor_reshaped, full_matrices=False)
    return U, S, Vh


@torch.no_grad()
def lowrank_svd(
    tensor_reshaped: Tensor,
    rank: int,
    n_iter: int = 2,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Low-rank randomized SVD baseline using torch.svd_lowrank.

    Approximates the leading `rank` singular triplets:
        A ≈ U @ diag(S) @ Vh

    Parameters
    ----------
    tensor_reshaped:
        Input matrix A with shape [..., m, n].
    rank:
        Target rank k for the approximation.
    n_iter:
        Number of subspace iterations used by torch.svd_lowrank.

    Returns
    -------
    U, S, Vh:
        U  : [..., m, k]
        S  : [..., k]
        Vh : [..., k, n]
    """
    U, S, V = torch.svd_lowrank(tensor_reshaped, q=rank, niter=n_iter)
    Vh = V.mH
    return U, S, Vh
