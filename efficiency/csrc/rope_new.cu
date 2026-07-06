/*
################################################################################
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################
*/


#include <torch/extension.h>
#include <cuda_bf16.h>
#include <vector>
#include "functions.h"

__global__ void apply_rotary_pos_emb_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ cos_sin,
    const int64_t* __restrict__ position_ids,
    __nv_bfloat16* __restrict__ output,
    int batch_size, int heads, int seq_len, int embed_dim,
    int stride_xb, int stride_xh, int stride_xs, int stride_xe,
    int stride_cos_sin,
    int stride_pid_b, int stride_pid_h, int stride_pid_s,
    int half_dim)
{
    int b_idx = blockIdx.x;
    int h_idx = blockIdx.y;
    int s_idx = blockIdx.z;
    int tid = threadIdx.x;

    int pid = position_ids[b_idx * stride_pid_b + h_idx * stride_pid_h + s_idx * stride_pid_s];
    const __nv_bfloat16* cos_sin_ptr = cos_sin + pid * stride_cos_sin;

    int x_offset = b_idx * stride_xb + h_idx * stride_xh + s_idx * stride_xs;
    const __nv_bfloat16* x_ptr = x + x_offset;
    __nv_bfloat16* output_ptr = output + x_offset;

    if (tid < half_dim) {
        __nv_bfloat16 x1 = x_ptr[tid];
        __nv_bfloat16 x2 = x_ptr[tid + half_dim];
        __nv_bfloat16 cos = cos_sin_ptr[tid];
        __nv_bfloat16 sin = cos_sin_ptr[tid + half_dim];

        output_ptr[tid] = __hadd(__hmul(x1, cos), __hmul(__hneg(x2), sin));
        output_ptr[tid + half_dim] = __hadd(__hmul(x2, cos), __hmul(x1, sin));
    }
}



void apply_rotary_pos_emb_new(
    torch::Tensor x, torch::Tensor cos_sin, torch::Tensor position_ids, torch::Tensor output,
    int64_t batch_size, int64_t heads, int64_t seq_len, int64_t embed_dim,
    int64_t stride_xb, int64_t stride_xh, int64_t stride_xs, int64_t stride_xe,
    int64_t stride_cos_sin,
    int64_t stride_pid_b, int64_t stride_pid_h, int64_t stride_pid_s,
    int64_t half_dim)
{
    const dim3 blocks(batch_size, heads, seq_len);
    const dim3 threads(half_dim);

    apply_rotary_pos_emb_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(cos_sin.data_ptr<at::BFloat16>()),
        position_ids.data_ptr<int64_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        (int)batch_size, (int)heads, (int)seq_len, (int)embed_dim,
        (int)stride_xb, (int)stride_xh, (int)stride_xs, (int)stride_xe,
        (int)stride_cos_sin,
        (int)stride_pid_b, (int)stride_pid_h, (int)stride_pid_s,
        (int)half_dim
    );
}

__global__ void apply_rotary_pos_emb_kernel_v2(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ cos_sin,
    const int32_t* __restrict__ position_ids,
    __nv_bfloat16* __restrict__ output,
    int batch_size, int heads, int seq_len, int embed_dim,
    int stride_xb, int stride_xh, int stride_xs, int stride_xe,
    int stride_cos_sin,
    int stride_pid_b, int stride_pid_h, int stride_pid_s,
    int half_dim, int chunk_size)
{
    int b_idx = blockIdx.x;
    int h_idx = blockIdx.y;
    int s_idx = blockIdx.z;
    int tid = threadIdx.x;

    int pid = position_ids[b_idx * stride_pid_b + h_idx * stride_pid_h + (s_idx / chunk_size) * stride_pid_s];
    const __nv_bfloat16* cos_sin_ptr = cos_sin + (pid * chunk_size + s_idx % chunk_size) * stride_cos_sin;

    int x_offset = b_idx * stride_xb + h_idx * stride_xh + s_idx * stride_xs;
    const __nv_bfloat16* x_ptr = x + x_offset;
    __nv_bfloat16* output_ptr = output + x_offset;

    if (tid < half_dim) {
        __nv_bfloat16 x1 = x_ptr[tid];
        __nv_bfloat16 x2 = x_ptr[tid + half_dim];
        __nv_bfloat16 cos = cos_sin_ptr[tid];
        __nv_bfloat16 sin = cos_sin_ptr[tid + half_dim];

        output_ptr[tid] = __hadd(__hmul(x1, cos), __hmul(__hneg(x2), sin));
        output_ptr[tid + half_dim] = __hadd(__hmul(x2, cos), __hmul(x1, sin));
    }
}



void apply_rotary_pos_emb_new_v2(
    torch::Tensor x, torch::Tensor cos_sin, torch::Tensor position_ids, torch::Tensor output,
    int64_t batch_size, int64_t heads, int64_t seq_len, int64_t embed_dim,
    int64_t stride_xb, int64_t stride_xh, int64_t stride_xs, int64_t stride_xe,
    int64_t stride_cos_sin,
    int64_t stride_pid_b, int64_t stride_pid_h, int64_t stride_pid_s,
    int64_t half_dim, int64_t chunk_size)
{
    const dim3 blocks(batch_size, heads, seq_len);
    const dim3 threads(half_dim);

    apply_rotary_pos_emb_kernel_v2<<<blocks, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(cos_sin.data_ptr<at::BFloat16>()),
        position_ids.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        (int)batch_size, (int)heads, (int)seq_len, (int)embed_dim,
        (int)stride_xb, (int)stride_xh, (int)stride_xs, (int)stride_xe,
        (int)stride_cos_sin,
        (int)stride_pid_b, (int)stride_pid_h, (int)stride_pid_s,
        (int)half_dim, (int)chunk_size
    );
}

__global__ void apply_rotary_pos_emb_kernel_push_cache(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ cos_sin,
    const int32_t* __restrict__ position_ids,
    __nv_bfloat16* __restrict__ output_cache,
    const int32_t* __restrict__ cnts,
    int batch_size, int heads, int seq_len, int embed_dim,
    int stride_xb, int stride_xh, int stride_xs, int stride_xe,
    int stride_cos_sin,
    int stride_pid_b, int stride_pid_h, int stride_pid_s,
    int stride_output_b, int stride_output_h, int stride_output_s,
    int offset_output_s_start, int offset_output_s_end,
    int half_dim, int chunk_size)
{
    int b_idx = blockIdx.x;
    int h_idx = blockIdx.y;
    int s_idx = blockIdx.z;
    int tid = threadIdx.x;

    int pid = position_ids[b_idx * stride_pid_b + h_idx * stride_pid_h + (s_idx / chunk_size) * stride_pid_s];
    int cnt = cnts[b_idx * heads + h_idx];
    if (s_idx / chunk_size < cnt) {
        return;
    }
    const __nv_bfloat16* cos_sin_ptr = cos_sin + (pid * chunk_size + s_idx % chunk_size) * stride_cos_sin;

    int x_offset = b_idx * stride_xb + h_idx * stride_xh + s_idx * stride_xs;
    int out_offset = b_idx * stride_output_b + h_idx * stride_output_h + (offset_output_s_start + s_idx) * stride_output_s;
    const __nv_bfloat16* x_ptr = x + x_offset;
    __nv_bfloat16* output_ptr = output_cache + out_offset;

    if (offset_output_s_start + s_idx >= offset_output_s_end) {
        return;
    }

    if (tid < half_dim) {
        __nv_bfloat16 x1 = x_ptr[tid];
        __nv_bfloat16 x2 = x_ptr[tid + half_dim];
        __nv_bfloat16 cos = cos_sin_ptr[tid];
        __nv_bfloat16 sin = cos_sin_ptr[tid + half_dim];

        output_ptr[tid] = __hadd(__hmul(x1, cos), __hmul(__hneg(x2), sin));
        output_ptr[tid + half_dim] = __hadd(__hmul(x2, cos), __hmul(x1, sin));
    }
}



void apply_rotary_pos_emb_push_cache(
    torch::Tensor x, torch::Tensor cos_sin, torch::Tensor position_ids, torch::Tensor output_cache,
    torch::Tensor cnts,
    int64_t batch_size, int64_t heads, int64_t seq_len, int64_t embed_dim,
    int64_t stride_xb, int64_t stride_xh, int64_t stride_xs, int64_t stride_xe,
    int64_t stride_cos_sin,
    int64_t stride_pid_b, int64_t stride_pid_h, int64_t stride_pid_s,
    int64_t stride_output_b, int64_t stride_output_h, int64_t stride_output_s,
    int64_t offset_output_s_start, int64_t offset_output_s_end,
    int64_t half_dim, int64_t chunk_size)
{
    const dim3 blocks(batch_size, heads, seq_len);
    const dim3 threads(half_dim);

    apply_rotary_pos_emb_kernel_push_cache<<<blocks, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(cos_sin.data_ptr<at::BFloat16>()),
        position_ids.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(output_cache.data_ptr<at::BFloat16>()),
        cnts.data_ptr<int32_t>(),
        (int)batch_size, (int)heads, (int)seq_len, (int)embed_dim,
        (int)stride_xb, (int)stride_xh, (int)stride_xs, (int)stride_xe,
        (int)stride_cos_sin,
        (int)stride_pid_b, (int)stride_pid_h, (int)stride_pid_s,
        (int)stride_output_b, (int)stride_output_h, (int)stride_output_s,
        (int)offset_output_s_start, (int)offset_output_s_end,
        (int)half_dim, (int)chunk_size
    );
}

__global__ void apply_rotary_pos_emb_kernel_push_cache_opt(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ cos_sin,
    const int32_t* __restrict__ position_ids,
    __nv_bfloat16* __restrict__ output_cache,
    const int32_t* __restrict__ cnts,
    int batch_size, int heads, int seq_len, int embed_dim,
    int stride_xb, int stride_xh, int stride_xs, int stride_xe,
    int stride_cos_sin,
    int stride_pid_b, int stride_pid_h, int stride_pid_s,
    int stride_output_b, int stride_output_h, int stride_output_s,
    int offset_output_s_start, int offset_output_s_end,
    int half_dim, int chunk_size)
{
    int h_idx = threadIdx.x / half_dim;
    int s_idx = blockIdx.x;
    int tid = threadIdx.x % half_dim;

    // In-kernel loop for batch size with unrolling
    #pragma unroll
    for (int b_idx = 0; b_idx < batch_size; b_idx++) {
        int cnt = cnts[b_idx * heads + h_idx];
        if (s_idx / chunk_size < cnt) {
            continue;
        }

        int pid = position_ids[b_idx * stride_pid_b + h_idx * stride_pid_h + (s_idx / chunk_size) * stride_pid_s];
        const __nv_bfloat16* cos_sin_ptr = cos_sin + (pid * chunk_size + s_idx % chunk_size) * stride_cos_sin;

        int x_offset = b_idx * stride_xb + h_idx * stride_xh + s_idx * stride_xs;
        int out_offset = b_idx * stride_output_b + h_idx * stride_output_h + (offset_output_s_start + s_idx) * stride_output_s;
        const __nv_bfloat16* x_ptr = x + x_offset;
        __nv_bfloat16* output_ptr = output_cache + out_offset;

        if (offset_output_s_start + s_idx >= offset_output_s_end) {
            return;
        }

        if (tid < half_dim) {
            __nv_bfloat16 x1 = x_ptr[tid];
            __nv_bfloat16 x2 = x_ptr[tid + half_dim];
            __nv_bfloat16 cos = cos_sin_ptr[tid];
            __nv_bfloat16 sin = cos_sin_ptr[tid + half_dim];

            output_ptr[tid] = __hadd(__hmul(x1, cos), __hmul(__hneg(x2), sin));
            output_ptr[tid + half_dim] = __hadd(__hmul(x2, cos), __hmul(x1, sin));
        }
    }
}



void apply_rotary_pos_emb_push_cache_opt(
    torch::Tensor x, torch::Tensor cos_sin, torch::Tensor position_ids, torch::Tensor output_cache,
    torch::Tensor cnts,
    int64_t batch_size, int64_t heads, int64_t seq_len, int64_t embed_dim,
    int64_t stride_xb, int64_t stride_xh, int64_t stride_xs, int64_t stride_xe,
    int64_t stride_cos_sin,
    int64_t stride_pid_b, int64_t stride_pid_h, int64_t stride_pid_s,
    int64_t stride_output_b, int64_t stride_output_h, int64_t stride_output_s,
    int64_t offset_output_s_start, int64_t offset_output_s_end,
    int64_t half_dim, int64_t chunk_size)
{

    apply_rotary_pos_emb_kernel_push_cache_opt<<<seq_len, heads * half_dim>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(cos_sin.data_ptr<at::BFloat16>()),
        position_ids.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(output_cache.data_ptr<at::BFloat16>()),
        cnts.data_ptr<int32_t>(),
        (int)batch_size, (int)heads, (int)seq_len, (int)embed_dim,
        (int)stride_xb, (int)stride_xh, (int)stride_xs, (int)stride_xe,
        (int)stride_cos_sin,
        (int)stride_pid_b, (int)stride_pid_h, (int)stride_pid_s,
        (int)stride_output_b, (int)stride_output_h, (int)stride_output_s,
        (int)offset_output_s_start, (int)offset_output_s_end,
        (int)half_dim, (int)chunk_size
    );
}


__global__ void apply_rotary_pos_emb_kernel_push_cache_opt_glm(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ cos_sin,
    const int32_t* __restrict__ position_ids,
    __nv_bfloat16* __restrict__ output_cache,
    const int32_t* __restrict__ cnts,
    int batch_size, int heads, int seq_len, int embed_dim,
    int stride_xb, int stride_xh, int stride_xs, int stride_xe,
    int stride_cos_sin,
    int stride_pid_b, int stride_pid_h, int stride_pid_s,
    int stride_output_b, int stride_output_h, int stride_output_s,
    int offset_output_s_start, int offset_output_s_end,
    int half_dim, int chunk_size)
{
    int h_idx = threadIdx.x / half_dim;
    int s_idx = blockIdx.x;

    // half dim is 64
    int tid = threadIdx.x % half_dim;

    // In-kernel loop for batch size with unrolling
    #pragma unroll
    for (int b_idx = 0; b_idx < batch_size; b_idx++) {
        int cnt = cnts[b_idx * heads + h_idx];
        if (s_idx / chunk_size < cnt) {
            continue;
        }

        int pid = position_ids[b_idx * stride_pid_b + h_idx * stride_pid_h + (s_idx / chunk_size) * stride_pid_s];
        const __nv_bfloat16* cos_sin_ptr = cos_sin + (pid * chunk_size + s_idx % chunk_size) * stride_cos_sin;

        int x_offset = b_idx * stride_xb + h_idx * stride_xh + s_idx * stride_xs;
        int out_offset = b_idx * stride_output_b + h_idx * stride_output_h + (offset_output_s_start + s_idx) * stride_output_s;
        const __nv_bfloat16* x_ptr = x + x_offset;
        __nv_bfloat16* output_ptr = output_cache + out_offset;

        if (offset_output_s_start + s_idx >= offset_output_s_end) {
            return;
        }

        // if (tid < half_dim) {
        //     __nv_bfloat16 x1 = x_ptr[tid];
        //     __nv_bfloat16 x2 = x_ptr[tid + half_dim];
        //     __nv_bfloat16 cos = cos_sin_ptr[tid];
        //     __nv_bfloat16 sin = cos_sin_ptr[tid + half_dim];

        //     output_ptr[tid] = __hadd(__hmul(x1, cos), __hmul(__hneg(x2), sin));
        //     output_ptr[tid + half_dim] = __hadd(__hmul(x2, cos), __hmul(x1, sin));
        // }
        if (tid < half_dim) {
            if (tid < 32) { // tid (0, 32)
                // First 32 dimensions
                int even_idx = tid * 2;
                int odd_idx = even_idx + 1;

                __nv_bfloat16 x1 = x_ptr[even_idx];
                __nv_bfloat16 x2 = x_ptr[odd_idx];
                __nv_bfloat16 cos = cos_sin_ptr[tid];
                __nv_bfloat16 sin = cos_sin_ptr[tid + 32];

                output_ptr[even_idx] = __hadd(__hmul(x1, cos), __hmul(__hneg(x2), sin));
                output_ptr[odd_idx] = __hadd(__hmul(x2, cos), __hmul(x1, sin));
            } else { // tid (32, 64)
                // Remaining dimensions (32 to 64) unchanged
                output_ptr[tid + 32] = x_ptr[tid + 32];
                output_ptr[tid + 64] = x_ptr[tid + 64];
            }
        }
    }
}



void apply_rotary_pos_emb_push_cache_opt_glm(
    torch::Tensor x, torch::Tensor cos_sin, torch::Tensor position_ids, torch::Tensor output_cache,
    torch::Tensor cnts,
    int64_t batch_size, int64_t heads, int64_t seq_len, int64_t embed_dim,
    int64_t stride_xb, int64_t stride_xh, int64_t stride_xs, int64_t stride_xe,
    int64_t stride_cos_sin,
    int64_t stride_pid_b, int64_t stride_pid_h, int64_t stride_pid_s,
    int64_t stride_output_b, int64_t stride_output_h, int64_t stride_output_s,
    int64_t offset_output_s_start, int64_t offset_output_s_end,
    int64_t half_dim, int64_t chunk_size)
{

    apply_rotary_pos_emb_kernel_push_cache_opt_glm<<<seq_len, heads * half_dim>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(cos_sin.data_ptr<at::BFloat16>()),
        position_ids.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(output_cache.data_ptr<at::BFloat16>()),
        cnts.data_ptr<int32_t>(),
        (int)batch_size, (int)heads, (int)seq_len, (int)embed_dim,
        (int)stride_xb, (int)stride_xh, (int)stride_xs, (int)stride_xe,
        (int)stride_cos_sin,
        (int)stride_pid_b, (int)stride_pid_h, (int)stride_pid_s,
        (int)stride_output_b, (int)stride_output_h, (int)stride_output_s,
        (int)offset_output_s_start, (int)offset_output_s_end,
        (int)half_dim, (int)chunk_size
    );
}
