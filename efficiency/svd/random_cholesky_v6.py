import torch
from torch import Tensor


PowerDType = str
OrthogonalizationMethod = str


@torch.no_grad()
def chol_qr(
    Y_low_precision: Tensor,
    eye: Tensor,
    base_eps: float = 1e-5,
    max_eps: float = 10.0,
    max_tries: int = 6,
    use_eigh_last: bool = True,
) -> Tensor:
    """
    Numerically robust Cholesky-QR for a batch of low-precision matrices.

    Given an input matrix (or batch of matrices) Y, this routine:
    1. Forms the Gram matrix G = Y^H Y in float32 and explicitly symmetrizes it.
    2. Adds an adaptive diagonal regularization eps * scale * I to improve
       positive-definiteness, where `scale` is derived from the mean diagonal of G.
    3. Attempts a Cholesky factorization with exponentially increasing eps
       until it succeeds or reaches `max_tries`.
    4. Optionally falls back to an eigendecomposition-based SPD repair if
       Cholesky keeps failing (controlled by `use_eigh_last`).
    5. Finally falls back to a standard QR on Y as a last resort.

    The output Q has the same dtype as the original input, but all internal
    computations are done in float32 for stability.
    """
    # Work in float32 for numerical stability.
    Y = Y_low_precision.float()

    # Gram matrix of Y: G = Y^H * Y. We symmetrize to avoid small Hermitian drift
    # introduced by finite precision arithmetic.
    G = torch.matmul(Y.mH, Y)
    G = 0.5 * (G + G.mH)

    # Compute a scaling factor based on the mean of the diagonal entries.
    # This scale is used to determine the magnitude of the diagonal jitter
    # we add to G to help enforce positive definiteness.
    d = torch.diagonal(G, dim1=-2, dim2=-1)
    scale = d.mean(dim=-1, keepdim=True).clamp_min_(1e-12).unsqueeze(-1)

    # Start from a small base regularization and increase it geometrically
    # if the Cholesky factorization fails.
    eps = base_eps

    for _ in range(max_tries):
        # Try a Cholesky factorization of G + eps * scale * I.
        R, info = torch.linalg.cholesky_ex(G + (eps * scale) * eye, upper=True)

        # If info == 0 everywhere, the factorization succeeded for all batches.
        if (info == 0).all():
            # Solve Q R = Y, equivalent to Q = Y R^{-1}.
            Q = torch.linalg.solve_triangular(R, Y, upper=True, left=False)
            return Q.to(Y_low_precision.dtype)

        # If Cholesky failed for some batches, increase eps and try again.
        eps = min(eps * 10.0, max_eps)

    # If repeated Cholesky attempts fail, we optionally try to repair G using
    # an eigendecomposition, clamp eigenvalues from below, and re‑Cholesky.
    if use_eigh_last:
        try:
            # Eigendecompose G, clamp eigenvalues to be at least max(1e-4, eps),
            # reconstruct a strictly SPD matrix G_spd, and factorize that.
            L, V = torch.linalg.eigh(G)
            L = torch.clamp(L, min=max(1e-4, eps))
            G_spd = torch.matmul(V, torch.matmul(torch.diag_embed(L), V.mH))
            R = torch.linalg.cholesky(G_spd, upper=True)
            Q = torch.linalg.solve_triangular(R, Y, upper=True, left=False)
            return Q.to(Y_low_precision.dtype)
        except Exception:
            # If anything goes wrong in the SPD repair path, silently fall back
            # to a standard QR factorization below.
            pass

    # Final fallback: run regular QR on Y. This is more expensive but robust.
    Q, _ = torch.linalg.qr(Y, mode="reduced")
    return Q.to(Y_low_precision.dtype)


@torch.no_grad()
def householder_qr(Y_low_precision: Tensor) -> Tensor:
    """
    Standard QR-based orthonormalization using Householder reflections.

    This is a simpler but typically more numerically stable alternative to
    `chol_qr`. It directly applies QR to Y in float32 and converts the result
    back to the input dtype.
    """
    Y = Y_low_precision.float()
    Q, _ = torch.linalg.qr(Y, mode="reduced")
    return Q.to(Y_low_precision.dtype)


@torch.no_grad()
def orthonormalize(
    Y_low_precision: Tensor,
    eye_q: Tensor,
    orth: OrthogonalizationMethod = "chol",
    base_eps: float = 1e-6,
    max_eps: float = 10.0,
    max_tries: int = 6,
    use_eigh_last: bool = True,
) -> Tensor:
    """
    Dispatch helper to orthonormalize the columns of a sketch matrix.

    Parameters
    ----------
    orth:
        - "chol": Use Cholesky–QR with adaptive diagonal regularization.
                  Faster and very memory‑friendly, but slightly less stable.
        - anything else (e.g. "house"): Use standard Householder QR, which is
          more stable but somewhat heavier.
    """
    if orth == "chol":
        return chol_qr(Y_low_precision, eye_q, base_eps, max_eps, max_tries, use_eigh_last)
    return householder_qr(Y_low_precision)


@torch.no_grad()
def randomized_svd_fp16(
    tensor_reshaped: Tensor,
    rank: int,
    n_iter: int = 4,
    oversample: int = 4,
    M: Tensor | None = None,
    base_eps: float = 1e-5,
    max_eps: float = 10.0,
    max_tries: int = 6,
    use_eigh_last: bool = True,
    power_dtype: PowerDType = "bf16",
    orth: OrthogonalizationMethod = "chol",
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Hardware-efficient randomized SVD for KV-cache compression.

    High-level algorithm:
    1. Optionally transpose wide matrices so that the working operator has
       more rows than columns (m >= n), which improves numerical behavior.
    2. Choose a working precision for the power iteration / projection
       (fp32, bf16, fp16, or fp8 variants).
    3. Build a low-dimensional random subspace that approximates the range of A.
    4. Apply subspace (power) iteration to sharpen separation between singular
       values and improve approximation quality.
    5. Project A to a small matrix B = Q^H A, compute an SVD of B.
    6. Lift the small SVD back to tall singular vectors of A.
    """
    # ------------------------------------------------------------------
    # 1) Basic shape handling and possible wide‑matrix transpose
    # ------------------------------------------------------------------
    m, n = tensor_reshaped.shape[-2:]
    batch_shape = tensor_reshaped.shape[:-2]

    # For numerical reasons it is usually better to work with an operator
    # that has at least as many rows as columns. If the input is "wide",
    # we conjugate‑transpose it and later undo this flip on the outputs.
    is_wide = m < n

    # ------------------------------------------------------------------
    # 2) Resolve low‑precision dtype used in power iteration / projection
    # ------------------------------------------------------------------
    # fp32 path:
    #   - use the full fp32 input
    #   - do not quantize to a lower precision
    #   - matches behavior / accuracy of torch.svd_lowrank
    #
    # low‑precision paths:
    #   - quantize the input once to a "low_dtype"
    #   - perform power iteration and projection in that low precision
    #   - significantly reduces memory footprint and bandwidth
    if power_dtype == "fp32":
        low_dtype = None
    elif power_dtype == "fp16":
        low_dtype = torch.float16
    elif power_dtype == "bf16":
        low_dtype = torch.bfloat16
    elif power_dtype in ("fp8", "fp8_e4m3"):
        if hasattr(torch, "float8_e4m3fn"):
            low_dtype = torch.float8_e4m3fn
        else:
            raise ValueError(
                "power_dtype='fp8' requested, but torch.float8_e4m3fn is not available in this PyTorch version."
            )
    elif power_dtype == "fp8_e5m2":
        if hasattr(torch, "float8_e5m2"):
            low_dtype = torch.float8_e5m2
        else:
            raise ValueError(
                "power_dtype='fp8_e5m2' requested, but torch.float8_e5m2 is not available in this PyTorch version."
            )
    else:
        raise ValueError(
            f"Unsupported power_dtype '{power_dtype}'. Use 'fp32', 'fp16', 'bf16', 'fp8', or 'fp8_e5m2'."
        )

    # ------------------------------------------------------------------
    # 3) Choose working representation X_pow (and optionally X_low)
    # ------------------------------------------------------------------
    if power_dtype == "fp32":
        X_pow = tensor_reshaped.float()
        if is_wide:
            X_pow = X_pow.mH
            if M is not None:
                M = M.mH
            m, n = n, m
        X_low = None  # not used in fp32 path
    else:
        X_low = tensor_reshaped.to(low_dtype)
        if is_wide:
            X_low = X_low.mH
            if M is not None:
                M = M.mH
            m, n = n, m
        X_pow = X_low

    # Target rank (k) is clipped by the effective matrix dimensions.
    k = min(rank, m, n)
    # q_actual is the dimension of the random subspace; oversampling by a small
    # amount (e.g. 5–10) helps capture slightly more of the spectrum and
    # improves the approximation of the top-k singular space.
    q_actual = min(k + oversample, m, n)

    # Optional mean / baseline tensor M that will be subtracted out. We keep
    # versions in both the power‑iteration dtype and the low‑precision dtype.
    if M is not None:
        M_pow = M.to(X_pow.dtype).broadcast_to(X_pow.shape)
        M_low = None if power_dtype == "fp32" else M.to(low_dtype).broadcast_to(X_low.shape)
    else:
        M_pow, M_low = None, None

    # The eye used inside orthonormalization should match the working dtype:
    # float32 for fp32 path, otherwise the chosen low precision.
    eye_dtype = torch.float32 if power_dtype == "fp32" else low_dtype
    eye_q = torch.eye(q_actual, device=tensor_reshaped.device, dtype=eye_dtype)

    # ------------------------------------------------------------------
    # 4) Initial random projection: Y = (A - M) * R
    # ------------------------------------------------------------------
    # Note: current PyTorch versions do not support randn directly in float8,
    # so for fp8 we first sample in at least float16 and then cast.
    rand_dtype = (
        torch.float16
        if power_dtype not in ("fp32", "fp16", "bf16")
        else X_pow.dtype
    )
    R_rand = torch.randn(
        *batch_shape,
        n,
        q_actual,
        device=tensor_reshaped.device,
        dtype=rand_dtype,
    ).to(X_pow.dtype)
    Y = torch.matmul(X_pow, R_rand)
    if M_pow is not None:
        Y = Y - torch.matmul(M_pow, R_rand)

    # For fp32 path we keep Y in float32 through orthonormalization so that we
    # closely match torch.svd_lowrank; for low‑precision paths we cast to
    # `low_dtype` to save memory and compute.
    Y_for_orth = Y if power_dtype == "fp32" else Y.to(low_dtype)
    Q = orthonormalize(Y_for_orth, eye_q, orth, base_eps, max_eps, max_tries, use_eigh_last)

    # ------------------------------------------------------------------
    # 5) Prepare transposed operator for power iteration
    # ------------------------------------------------------------------
    X_t = X_pow.mH
    M_t = M_pow.mH if M_pow is not None else None

    # ------------------------------------------------------------------
    # 6) Power iteration (subspace iteration)
    #    Repeatedly apply (A^H A) to refine Q and improve spectral separation.
    # ------------------------------------------------------------------
    for _ in range(n_iter):
        # Y = (A^H - M^H) * Q
        Y = torch.matmul(X_t, Q.to(X_pow.dtype))
        if M_t is not None:
            Y = Y - torch.matmul(M_t, Q.to(X_pow.dtype))

        # Re‑orthonormalize to keep the basis well conditioned.
        Y_for_orth = Y if power_dtype == "fp32" else Y.to(low_dtype)
        Q = orthonormalize(Y_for_orth, eye_q, orth, base_eps, max_eps, max_tries, use_eigh_last)

        # Y = (A - M) * Q
        Y = torch.matmul(X_pow, Q.to(X_pow.dtype))
        if M_pow is not None:
            Y = Y - torch.matmul(M_pow, Q.to(X_pow.dtype))

        # Orthonormalize again to obtain the updated subspace Q.
        Y_for_orth = Y if power_dtype == "fp32" else Y.to(low_dtype)
        Q = orthonormalize(Y_for_orth, eye_q, orth, base_eps, max_eps, max_tries, use_eigh_last)

    # ------------------------------------------------------------------
    # 7) Projection: B = Q^H (A - M)
    # ------------------------------------------------------------------
    X_proj = X_pow if power_dtype == "fp32" else X_low
    M_proj = M_pow if power_dtype == "fp32" else M_low
    Q_proj = Q.to(X_proj.dtype)  # match dtype of the projected operator
    Bproj = torch.matmul(Q_proj.mH, X_proj)
    if M_proj is not None:
        Bproj = Bproj - torch.matmul(Q_proj.mH, M_proj)

    # ------------------------------------------------------------------
    # 8) Small SVD on B
    # ------------------------------------------------------------------
    # We compute SVD in float32 regardless of the power‑iteration dtype for
    # accuracy, then cast down later if needed.
    U_small, S, Vh_small = torch.linalg.svd(Bproj.float(), full_matrices=False)

    # ------------------------------------------------------------------
    # 9) Lift back to tall singular vectors and cast dtypes
    # ------------------------------------------------------------------
    # fp32 path: keep everything in fp32 (no low‑precision).
    # low‑precision paths: cast the outputs to `low_dtype` for downstream use.
    U_tall = torch.matmul(Q, U_small.to(Q.dtype))
    Vh_tall = Vh_small.to(Q.dtype)
    S = S.to(Q.dtype)
    if power_dtype != "fp32":
        U_tall = U_tall.to(low_dtype)
        Vh_tall = Vh_tall.to(low_dtype)
        S = S.to(low_dtype)

    # ------------------------------------------------------------------
    # 10) Final shape handling
    # ------------------------------------------------------------------
    # At this point U_tall, S, Vh_tall correspond to the (possibly transposed)
    # operator. We:
    #   - truncate to the requested rank
    #   - undo the "wide" transpose if we flipped the operator at the start.
    r = min(rank, S.shape[-1])
    
    if is_wide:
        U_orig = Vh_tall.mH
        Vh_orig = U_tall.mH
        return U_orig[..., :, :r], S[..., :r], Vh_orig[..., :r, :]
    else:
        return U_tall[..., :, :r], S[..., :r], Vh_tall[..., :r, :]
