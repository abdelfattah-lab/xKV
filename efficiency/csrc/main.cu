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


#include <Python.h>
#include <torch/library.h>
#include "functions.h"

// Define operator schemas using TORCH_LIBRARY
// This integrates better with PyTorch subsystems like torch.compile, autograd, etc.
TORCH_LIBRARY(_shadowkv, m) {
    // Gather-Copy operations
    m.def("gather_copy(Tensor values, Tensor(a!) v_cache_buffer, Tensor position_ids, "
          "int batch_size, int heads, int cpu_v_length, int gpu_v_length, int map_size) -> ()");
    
    m.def("gather_copy_d2d_with_offsets(Tensor(a!) keys, Tensor offsets, Tensor cnts, "
          "int batch_size, int heads, int gpu_k_length, int gpu_k_offset, int gpu_k_stride, int map_size) -> ()");
    
    m.def("reorder_keys_and_compute_offsets(Tensor(a!) cached_pos_ids, Tensor cur_pos_ids, "
          "Tensor(b!) offsets, Tensor(c!) cnts, int batch_size, int heads, int map_size) -> ()");
    
    m.def("gather_copy_with_offsets(Tensor values, Tensor(a!) v_cache_buffer, Tensor(b!) temp, "
          "Tensor offsets, Tensor cnts, Tensor(c!) signals, "
          "int batch_size, int heads, int cpu_v_length, int gpu_v_length, "
          "int gpu_v_offset, int gpu_v_stride, int map_size) -> ()");
    
    // Rotary Position Embedding operations
    m.def("apply_rotary_pos_emb(Tensor x, Tensor cos, Tensor sin, Tensor position_ids, Tensor(a!) output, "
          "int batch_size, int heads, int seq_len, int embed_dim, "
          "int stride_xb, int stride_xh, int stride_xs, int stride_xe, "
          "int stride_cos, int stride_sin, "
          "int stride_pid_b, int stride_pid_h, int stride_pid_s, "
          "int half_dim) -> ()");
    
    m.def("apply_rotary_pos_emb_new(Tensor x, Tensor cos_sin, Tensor position_ids, Tensor(a!) output, "
          "int batch_size, int heads, int seq_len, int embed_dim, "
          "int stride_xb, int stride_xh, int stride_xs, int stride_xe, "
          "int stride_cos_sin, "
          "int stride_pid_b, int stride_pid_h, int stride_pid_s, "
          "int half_dim) -> ()");
    
    m.def("apply_rotary_pos_emb_new_v2(Tensor x, Tensor cos_sin, Tensor position_ids, Tensor(a!) output, "
          "int batch_size, int heads, int seq_len, int embed_dim, "
          "int stride_xb, int stride_xh, int stride_xs, int stride_xe, "
          "int stride_cos_sin, "
          "int stride_pid_b, int stride_pid_h, int stride_pid_s, "
          "int half_dim, int chunk_size) -> ()");
    
    m.def("apply_rotary_pos_emb_push_cache(Tensor x, Tensor cos_sin, Tensor position_ids, Tensor(a!) output, "
          "Tensor cnts, "
          "int batch_size, int heads, int seq_len, int embed_dim, "
          "int stride_xb, int stride_xh, int stride_xs, int stride_xe, "
          "int stride_cos_sin, "
          "int stride_pid_b, int stride_pid_h, int stride_pid_s, "
          "int stride_output_b, int stride_output_h, int stride_output_s, "
          "int offset_output_s_start, int offset_output_s_end, "
          "int half_dim, int chunk_size) -> ()");
    
    m.def("apply_rotary_pos_emb_push_cache_opt(Tensor x, Tensor cos_sin, Tensor position_ids, Tensor(a!) output, "
          "Tensor cnts, "
          "int batch_size, int heads, int seq_len, int embed_dim, "
          "int stride_xb, int stride_xh, int stride_xs, int stride_xe, "
          "int stride_cos_sin, "
          "int stride_pid_b, int stride_pid_h, int stride_pid_s, "
          "int stride_output_b, int stride_output_h, int stride_output_s, "
          "int offset_output_s_start, int offset_output_s_end, "
          "int half_dim, int chunk_size) -> ()");
    
    m.def("apply_rotary_pos_emb_push_cache_opt_glm(Tensor x, Tensor cos_sin, Tensor position_ids, Tensor(a!) output, "
          "Tensor cnts, "
          "int batch_size, int heads, int seq_len, int embed_dim, "
          "int stride_xb, int stride_xh, int stride_xs, int stride_xe, "
          "int stride_cos_sin, "
          "int stride_pid_b, int stride_pid_h, int stride_pid_s, "
          "int stride_output_b, int stride_output_h, int stride_output_s, "
          "int offset_output_s_start, int offset_output_s_end, "
          "int half_dim, int chunk_size) -> ()");
    
    // GEMM operations
    m.def("batch_gather_gemm(Tensor a, Tensor b, Tensor cos, Tensor sin, Tensor position_ids, Tensor(a!) output, "
          "int batch_size, int heads, int seq_len, int embed_dim, int rank, int sparse_budget, "
          "int max_seq_len, int chunk_size, Tensor offset_array) -> ()");
    
    m.def("batch_gemm_softmax(Tensor A, Tensor B, Tensor(a!) D, Tensor(b!) Norm, Tensor(c!) Sum, "
          "Tensor(d!) Softmax, int batch_count, int m, int n, int k, float alpha, float beta) -> ()");
}

// Register CUDA implementations
TORCH_LIBRARY_IMPL(_shadowkv, CUDA, m) {
    m.impl("gather_copy", &gather_copy);
    m.impl("gather_copy_d2d_with_offsets", &gather_copy_d2d_with_offsets);
    m.impl("reorder_keys_and_compute_offsets", &reorder_keys_and_compute_offsets);
    m.impl("gather_copy_with_offsets", &gather_copy_with_offsets);
    m.impl("apply_rotary_pos_emb", &apply_rotary_pos_emb);
    m.impl("apply_rotary_pos_emb_new", &apply_rotary_pos_emb_new);
    m.impl("apply_rotary_pos_emb_new_v2", &apply_rotary_pos_emb_new_v2);
    m.impl("apply_rotary_pos_emb_push_cache", &apply_rotary_pos_emb_push_cache);
    m.impl("apply_rotary_pos_emb_push_cache_opt", &apply_rotary_pos_emb_push_cache_opt);
    m.impl("apply_rotary_pos_emb_push_cache_opt_glm", &apply_rotary_pos_emb_push_cache_opt_glm);
    m.impl("batch_gather_gemm", &batch_gather_gemm);
    m.impl("batch_gemm_softmax", &batch_gemm_softmax);
}

// Dummy PyInit function to make the module importable as a Python extension
// This ensures the TORCH_LIBRARY static initializers run when the module is loaded
static PyModuleDef shadowkv_module = {
    PyModuleDef_HEAD_INIT,
    "_shadowkv",  // name
    nullptr,       // doc
    -1,           // size
    nullptr,      // methods
    nullptr,      // slots
    nullptr,      // traverse
    nullptr,      // clear
    nullptr       // free
};

extern "C" {
    PyObject* PyInit__shadowkv(void) {
        // Return a dummy module - the actual operators are registered via TORCH_LIBRARY
        return PyModule_Create(&shadowkv_module);
    }
}