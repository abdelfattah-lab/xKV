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
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    const int64_t* __restrict__ position_ids,
    __nv_bfloat16* __restrict__ output,
    int batch_size, int heads, int seq_len, int embed_dim,
    int stride_xb, int stride_xh, int stride_xs, int stride_xe,
    int stride_cos, int stride_sin,
    int stride_pid_b, int stride_pid_h, int stride_pid_s,
    int half_dim)
{
    int b_idx = blockIdx.x;
    int h_idx = blockIdx.y;
    int s_idx = blockIdx.z;
    int tid = threadIdx.x;

    int pid = position_ids[b_idx * stride_pid_b + h_idx * stride_pid_h + s_idx * stride_pid_s];
    const __nv_bfloat16* cos_ptr = cos + pid * stride_cos;
    const __nv_bfloat16* sin_ptr = sin + pid * stride_sin;

    int x_offset = b_idx * stride_xb + h_idx * stride_xh + s_idx * stride_xs;
    const __nv_bfloat16* x_ptr = x + x_offset;
    __nv_bfloat16* output_ptr = output + x_offset;

    if (tid < half_dim) {
        __nv_bfloat16 x1 = x_ptr[tid];
        __nv_bfloat16 x2 = x_ptr[tid + half_dim];
        __nv_bfloat16 cos1 = cos_ptr[tid];
        __nv_bfloat16 sin1 = sin_ptr[tid];
        __nv_bfloat16 cos2 = cos_ptr[tid + half_dim];
        __nv_bfloat16 sin2 = sin_ptr[tid + half_dim];

        output_ptr[tid] = __hadd(__hmul(x1, cos1), __hmul(__hneg(x2), sin1));
        output_ptr[tid + half_dim] = __hadd(__hmul(x2, cos2), __hmul(x1, sin2));
    }
}

void apply_rotary_pos_emb(
    torch::Tensor x, torch::Tensor cos, torch::Tensor sin, torch::Tensor position_ids, torch::Tensor output,
    int64_t batch_size, int64_t heads, int64_t seq_len, int64_t embed_dim,
    int64_t stride_xb, int64_t stride_xh, int64_t stride_xs, int64_t stride_xe,
    int64_t stride_cos, int64_t stride_sin,
    int64_t stride_pid_b, int64_t stride_pid_h, int64_t stride_pid_s,
    int64_t half_dim)
{
    const dim3 blocks(batch_size, heads, seq_len);
    const dim3 threads(half_dim);

    apply_rotary_pos_emb_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(cos.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(sin.data_ptr<at::BFloat16>()),
        position_ids.data_ptr<int64_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        (int)batch_size, (int)heads, (int)seq_len, (int)embed_dim,
        (int)stride_xb, (int)stride_xh, (int)stride_xs, (int)stride_xe,
        (int)stride_cos, (int)stride_sin,
        (int)stride_pid_b, (int)stride_pid_h, (int)stride_pid_s,
        (int)half_dim
    );
}
