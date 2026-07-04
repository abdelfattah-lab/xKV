"""
Fused decode attention with RoPE on keys.

Fused decode attention kernel with RoPE applied to reconstructed K in Stage 1.

RoPE strategy (identical to qAB/v3):
  Split head_dim into two halves (each DIM_H = head_dim // 2 = 64).
  In the rank-accumulation loop, load k_B in two halves and keep two
  partial-accumulator tensors k_0 and k_1 (each [DIM_H, BLOCK_N]).
  After the loop apply the standard RoPE rotation per KV position:
    k_0_pe = k_0 * cos - k_1 * sin
    k_1_pe = k_1 * cos + k_0 * sin
  Then compute:
    qk = q_0 @ k_0_pe + q_1 @ k_1_pe

Stage 2 (combine splits) and Stage 3 (multiply by v_B) follow the same structure as Stage 1.

Interface:
  decode_attention_fwd(q, k_A_buffer, k_B_buffer, v_A_buffer, v_B_buffer,
                       num_kv_splits, max_kv_splits, sm_scale=1.0, theta=10_000_000.0)
  q:            [batch, num_q_heads, head_dim]
  k_A_buffer:   [batch, kv_len,      rank_k]
  k_B_buffer:   [batch, num_kv_heads, rank_k, head_dim]
  v_A_buffer:   [batch, kv_len,      rank_v]
  v_B_buffer:   [batch, num_kv_heads, rank_v, head_dim]
"""

import logging

import triton
import triton.language as tl
import torch

logger = logging.getLogger(__name__)

logger.debug(
    "The following error message 'operation scheduled before its operands' can be ignored."
)

_MIN_BLOCK_KV = 32


# ── RoPE sin/cos helper (inlined from ops/rope/sin_cos.py) ───────────────────

@triton.jit
def _sin_cos(starting_idx, theta: tl.constexpr, NB_TOKENS: tl.constexpr):
    DIM: tl.constexpr   = 128
    DIM_2: tl.constexpr = 64
    freqs = tl.arange(0, DIM_2) * 2
    freqs = freqs.to(tl.float32) / DIM
    freqs = tl.extra.cuda.libdevice.fast_powf(theta, freqs)
    freqs = (tl.arange(0, NB_TOKENS) + starting_idx)[:, None] / freqs[None, :]
    return (
        tl.extra.cuda.libdevice.fast_cosf(freqs),
        tl.extra.cuda.libdevice.fast_sinf(freqs),
    )


# ── Stage 1: compute partial attention per KV split (with RoPE on K) ─────────

@triton.jit
def _fwd_grouped_kernel_stage1_rope(
    Q,
    K_A_Buffer,   # [batch, kv_len, rank_k]
    K_B_Buffer,   # [batch, num_kv_heads, rank_k, head_dim]
    V_A_Buffer,   # [batch, kv_len, rank_v]
    sm_scale,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kA_bs,
    stride_buf_kA_ls,
    stride_buf_kB_bs,
    stride_buf_kB_hs,
    stride_buf_kB_rs,
    stride_buf_vbs,
    stride_buf_vls,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_seq_len:   tl.constexpr,
    kv_group_num: tl.constexpr,
    rank_k:       tl.constexpr,
    q_head_num:   tl.constexpr,
    BLOCK_DH:     tl.constexpr,   # head_dim // 2  (= 64 for Llama)
    BLOCK_DV:     tl.constexpr,   # next_power_of_2(rank_v)
    BLOCK_N:      tl.constexpr,   # tokens per KV tile
    BLOCK_H:      tl.constexpr,   # Q-heads per program
    BLOCK_R:      tl.constexpr,   # rank chunk size
    MIN_BLOCK_KV: tl.constexpr,
    Lk:           tl.constexpr,   # actual head_dim (= 2 * BLOCK_DH)
    Lv:           tl.constexpr,   # actual rank_v
    THETA:        tl.constexpr,   # RoPE base frequency
):
    cur_batch    = tl.program_id(0)
    cur_head_id  = tl.program_id(1)
    split_kv_id  = tl.program_id(2)
    cur_kv_head  = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h   = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h   = mask_h & (cur_head < q_head_num)

    offs_dh = tl.arange(0, BLOCK_DH)
    offs_r  = tl.arange(0, BLOCK_R)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_dv = offs_dv < Lv

    cur_kv_seq_len = kv_seq_len
    kv_splits = tl.load(num_kv_splits + cur_batch)

    # ── Load Q (two halves) ──────────────────────────────────────────────────
    offs_q_base = cur_batch * stride_qbs + cur_head[:, None] * stride_qh
    q_0 = tl.load(Q + offs_q_base + offs_dh[None, :],             mask=mask_h[:, None], other=0.0)
    q_1 = tl.load(Q + offs_q_base + (offs_dh + BLOCK_DH)[None, :], mask=mask_h[:, None], other=0.0)

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_kv_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end   = tl.minimum(split_kv_start + kv_len_per_split, cur_kv_seq_len)

    partial_qk_max  = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    partial_exp_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc             = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < split_kv_end

            # ── Reconstruct K in two halves ──────────────────────────────────
            k_0 = tl.zeros((BLOCK_DH, BLOCK_N), dtype=tl.float32)
            k_1 = tl.zeros((BLOCK_DH, BLOCK_N), dtype=tl.float32)

            for i in range(0, tl.cdiv(rank_k, BLOCK_R)):
                r_off = i * BLOCK_R

                offs_buf_kA = (
                    cur_batch * stride_buf_kA_bs
                    + offs_n[None, :] * stride_buf_kA_ls
                    + (offs_r + r_off)[:, None]
                )
                kA = tl.load(K_A_Buffer + offs_buf_kA, mask=mask_n[None, :], other=0.0)

                offs_buf_kB_0 = (
                    cur_batch * stride_buf_kB_bs
                    + cur_kv_head * stride_buf_kB_hs
                    + (offs_r + r_off)[None, :] * stride_buf_kB_rs
                    + offs_dh[:, None]
                )
                kB_0 = tl.load(K_B_Buffer + offs_buf_kB_0)

                offs_buf_kB_1 = (
                    cur_batch * stride_buf_kB_bs
                    + cur_kv_head * stride_buf_kB_hs
                    + (offs_r + r_off)[None, :] * stride_buf_kB_rs
                    + (offs_dh + BLOCK_DH)[:, None]
                )
                kB_1 = tl.load(K_B_Buffer + offs_buf_kB_1)

                k_0 = tl.dot(kB_0, kA, k_0)
                k_1 = tl.dot(kB_1, kA, k_1)

            # ── Apply RoPE ───────────────────────────────────────────────────
            cos, sin = _sin_cos(starting_idx=start_n, theta=THETA, NB_TOKENS=BLOCK_N)
            k_0_pe = k_0 * cos.T - k_1 * sin.T
            k_1_pe = k_1 * cos.T + k_0 * sin.T

            # ── Attention scores ─────────────────────────────────────────────
            qk  = tl.dot(q_0, k_0_pe.to(q_0.dtype))
            qk += tl.dot(q_1, k_1_pe.to(q_1.dtype))
            qk  = qk * sm_scale
            qk  = tl.where(mask_h[:, None] & mask_n[None, :], qk, float("-inf"))

            # ── V accumulation ───────────────────────────────────────────────
            offs_buf_v = (
                cur_batch * stride_buf_vbs
                + offs_n[:, None] * stride_buf_vls
                + offs_dv[None, :]
            )
            v = tl.load(V_A_Buffer + offs_buf_v, mask=mask_n[:, None] & mask_dv[None, :], other=0.0)

            new_partial_qk_max = tl.maximum(tl.max(qk, 1), partial_qk_max)
            re_scale = tl.exp(partial_qk_max - new_partial_qk_max)
            p = tl.exp(qk - new_partial_qk_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)

            partial_exp_sum = partial_exp_sum * re_scale + tl.sum(p, 1)
            partial_qk_max  = new_partial_qk_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv[None, :]
        )
        tl.store(Att_Out + offs_mid_o, acc / partial_exp_sum[:, None],
                 mask=mask_h[:, None] & mask_dv[None, :])

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv
        tl.store(Att_Lse + offs_mid_o_1, partial_qk_max + tl.log(partial_exp_sum), mask=mask_h)


# ── Stage 2: combine KV splits ────────────────────────────────────────────────

@triton.jit
def _fwd_kernel_stage2(
    attn_logits,
    attn_lse,
    o,
    num_kv_splits,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    kv_seq_len:    tl.constexpr,
    MAX_KV_SPLITS: tl.constexpr,
    MIN_BLOCK_KV:  tl.constexpr,
    BLOCK_DV:      tl.constexpr,
    Lv:            tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head  = tl.program_id(1)

    cur_kv_seq_len = kv_seq_len
    kv_splits = tl.load(num_kv_splits + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    global_exp_sum = 0.0
    global_lse     = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v     = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh) // Lv
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_kv_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )

    for split_kv_id in range(0, MAX_KV_SPLITS):
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end   = tl.minimum(split_kv_start + kv_len_per_split, cur_kv_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(attn_logits + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0)
            in_coming_lse = tl.load(attn_lse + offs_logic + split_kv_id * stride_mid_os // Lv)
            new_global_lse = tl.maximum(in_coming_lse, global_lse)
            old_scale = tl.exp(global_lse - new_global_lse)
            acc *= old_scale
            exp_logic = tl.exp(in_coming_lse - new_global_lse)
            acc += exp_logic * tv
            global_exp_sum = global_exp_sum * old_scale + exp_logic
            global_lse = new_global_lse

    tl.store(
        o + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / global_exp_sum,
        mask=mask_d,
    )


def _decode_softmax_reducev_fwd(attn_logits, attn_lse, q, o, v_A_buffer, num_kv_splits, max_kv_splits):
    batch, head_num = q.shape[0], q.shape[1]
    Lv         = v_A_buffer.shape[-1]
    kv_seq_len = v_A_buffer.shape[1]
    BLOCK_DV   = triton.next_power_of_2(Lv)

    _fwd_kernel_stage2[(batch, head_num)](
        attn_logits, attn_lse, o, num_kv_splits,
        attn_logits.stride(0), attn_logits.stride(1), attn_logits.stride(2),
        o.stride(0), o.stride(1),
        kv_seq_len=kv_seq_len,
        MAX_KV_SPLITS=max_kv_splits,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        num_warps=4,
        num_stages=2,
    )


# ── Stage 1 launcher ─────────────────────────────────────────────────────────

def _decode_grouped_att_m_fwd_rope(q, k_A_buffer, k_B_buffer, v_A_buffer,
                                   att_out, att_lse, num_kv_splits, max_kv_splits,
                                   sm_scale, theta):
    BLOCK_N = 32
    BLOCK_R = 64
    BLOCK_H = 16

    Dk = k_B_buffer.shape[-1]
    Dv = v_A_buffer.shape[-1]

    assert Dk % 2 == 0, "head_dim must be even for RoPE"
    BLOCK_DH = Dk // 2
    BLOCK_DV = triton.next_power_of_2(Dv)

    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num    = q.shape[1] // k_B_buffer.shape[1]
    kv_len          = k_A_buffer.shape[1]
    rk              = k_A_buffer.shape[-1]

    grid = (batch, triton.cdiv(head_num, min(BLOCK_H, kv_group_num)), max_kv_splits)

    _fwd_grouped_kernel_stage1_rope[grid](
        q, k_A_buffer, k_B_buffer, v_A_buffer,
        sm_scale, att_out, att_lse, num_kv_splits,
        q.stride(0), q.stride(1),
        k_A_buffer.stride(0), k_A_buffer.stride(1),
        k_B_buffer.stride(0), k_B_buffer.stride(1), k_B_buffer.stride(2),
        v_A_buffer.stride(0), v_A_buffer.stride(1),
        att_out.stride(0), att_out.stride(1), att_out.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        kv_seq_len=kv_len,
        rank_k=rk,
        BLOCK_DH=BLOCK_DH,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK_N,
        BLOCK_H=BLOCK_H,
        BLOCK_R=BLOCK_R,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        Lk=Dk,
        Lv=Dv,
        THETA=theta,
        num_warps=4,
        num_stages=2,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def decode_attention_fwd(
    q,              # [batch, num_q_heads, head_dim]
    k_A_buffer,    # [batch, kv_len,       rank_k]
    k_B_buffer,    # [batch, num_kv_heads,  rank_k, head_dim]
    v_A_buffer,    # [batch, kv_len,        rank_v]
    v_B_buffer,    # [batch, num_kv_heads,  rank_v, head_dim]
    num_kv_splits, # [batch]
    max_kv_splits, # int
    sm_scale=1.0,
    theta=10_000_000.0,
):
    """
    Full decode attention with RoPE applied to reconstructed K.

    theta (float): RoPE base frequency (default 10_000_000 for Llama-3).
    """
    assert v_A_buffer.shape[1] == k_A_buffer.shape[1]
    assert v_B_buffer.shape[1] == k_B_buffer.shape[1]

    num_kv_heads = v_B_buffer.shape[1]
    kv_group_num = q.shape[1] // num_kv_heads
    assert kv_group_num >= 1

    batch, num_q_heads, head_dim = q.shape
    rank_v = v_A_buffer.shape[-1]

    attn_logits = torch.empty(
        (batch, num_q_heads, max_kv_splits, rank_v), dtype=torch.float32, device=q.device,
    )
    attn_lse = torch.empty(
        (batch, num_q_heads, max_kv_splits), dtype=torch.float32, device=q.device,
    )

    _decode_grouped_att_m_fwd_rope(
        q, k_A_buffer, k_B_buffer, v_A_buffer,
        attn_logits, attn_lse,
        num_kv_splits, max_kv_splits,
        sm_scale, theta,
    )

    o = torch.empty(batch, num_q_heads, rank_v, device=q.device, dtype=q.dtype)
    _decode_softmax_reducev_fwd(attn_logits, attn_lse, q, o, v_A_buffer, num_kv_splits, max_kv_splits)

    # Stage 3: o @ v_B  →  [batch, num_q_heads, head_dim]
    o = torch.matmul(
        o.reshape(batch, num_kv_heads, -1, rank_v),
        v_B_buffer,
    ).reshape(batch, num_q_heads, head_dim)

    return o
