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

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <fstream>
#include <random>
#include <numeric>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "helper.h"

#include "gemm_universal_batch_gather_indices.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "functions.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = float;                  // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator; // <- data type of epilogue operations
using ElementInputA = cutlass::bfloat16_t;         // <- data type of elements in input matrix A
using ElementInputB = cutlass::bfloat16_t;         // <- data type of elements in input matrix B
using ElementOutput = cutlass::bfloat16_t;                       // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices.
// Column Major for Matrix A, B and C.
//
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

// This code section describes the tile size a thread block will compute
static const int ShapeMMAThreadBlockN = 128;
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, ShapeMMAThreadBlockN, 32>; // <- threadblock tile M = 128, N = 128, K = 32
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>; // <- warp tile M = 64, N = 64, K = 32
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>; // <- MMA Op tile M = 8, N = 8, K = 4
// 16, 8, 8 -> Turing
// 16, 8, 16 -> Ampere

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// Define the epilogue operation as LinearCombination. This is approximately equal to
//
//    d_ij = alpha * sum_k(a_ik * b_kj) + c_ij
//
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                    // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value, // <- this is the number of elements per
                                                      // vectorized memory access. For half
                                                      // precision, it's 8 elements. This becomes
                                                      // the vector width of math instructions in
                                                      // epilogue too
    ElementAccumulator,                               // <- data type of accumulator
    ElementComputeEpilogue>;                          // <- data type for alpha in linear combination function

// Number of pipelines you want to use
constexpr int NumStages = 5;
// Ampere -> 4/5
// Turing -> 2

using Gemm = cutlass::gemm::device::GemmUniversalBatchGatherIndices<
    ElementInputA,
    LayoutInputA,
    ElementInputB,
    LayoutInputB,
    ElementOutput,
    LayoutOutput,
    ElementAccumulator,
    MMAOp,
    SmArch,
    ShapeMMAThreadBlock,
    ShapeMMAWarp,
    ShapeMMAOp,
    EpilogueOp,
    SwizzleThreadBlock,
    NumStages,
    8, /*alignmentA*/
    8, /*alignmentB*/
    cutlass::arch::OpMultiplyAdd,
    cutlass::ComplexTransform::kNone,
    cutlass::ComplexTransform::kNone,
    true,  /*GatherA*/
    false, /*GatherB*/
    false  /*ScatterD*/
    >;

void batch_gather_gemm(
    torch::Tensor a, torch::Tensor b,
    torch::Tensor cos, torch::Tensor sin,
    torch::Tensor position_ids,
    torch::Tensor output,
    int64_t batch_size, int64_t heads, int64_t seq_len, int64_t embed_dim, int64_t rank, int64_t sparse_budget,
    int64_t max_seq_len, int64_t chunk_size, torch::Tensor offset_array)
{

    // Initialize alpha/beta for dot product computation
    ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
    ElementComputeEpilogue beta = ElementComputeEpilogue(0);

    cutlass::gemm::GemmCoord problem_size = {(int)seq_len, (int)embed_dim, (int)rank};

    cutlass::gemm::GemmCoord problem_size_real((int)sparse_budget,
                                               problem_size.n(),
                                               problem_size.k());

    // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
    // instantiated CUTLASS kernel
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kBatched,
        problem_size_real,                                                   // <- problem size of matrix multiplication
        (int)(batch_size * heads),                                                          // <- batch_size
        {alpha, beta},                                                       // <- alpha, beta
        reinterpret_cast<ElementInputA *>(a.data_ptr<at::BFloat16>()),      // <- reference to matrix A on device
        reinterpret_cast<ElementInputB *>(b.data_ptr<at::BFloat16>()),      // <- reference to matrix B on device
        reinterpret_cast<ElementOutput *>(output.data_ptr<at::BFloat16>()), // <- reference to matrix C on device
        reinterpret_cast<ElementOutput *>(output.data_ptr<at::BFloat16>()),
        reinterpret_cast<ElementOutput *>(sin.data_ptr<at::BFloat16>()),
        reinterpret_cast<ElementOutput *>(cos.data_ptr<at::BFloat16>()),
        (int64_t)problem_size.m() * problem_size.k(),
        (int64_t)problem_size.n() * problem_size.k(),
        (int64_t)sparse_budget * problem_size.n(),
        (int64_t)sparse_budget * problem_size.n(),
        LayoutInputA::packed(problem_size.mk()).stride(),
        LayoutInputB::packed(problem_size.kn()).stride(),
        LayoutOutput::packed({(int)sparse_budget, problem_size.n()}).stride(),
        LayoutOutput::packed({(int)sparse_budget, problem_size.n()}).stride(),
        LayoutOutput::packed({(int)max_seq_len, problem_size.n()}).stride(),
        LayoutOutput::packed({(int)max_seq_len, problem_size.n()}).stride(),
        reinterpret_cast<int *>(position_ids.data_ptr<int>()), // <- pointer to index vector to gather A on device
        nullptr,                                               // <- pointer to index vector to gather B on device
        nullptr,                                               // scatter D
        reinterpret_cast<int *>(position_ids.data_ptr<int>()), // <- pointer to index vector to gather sin_cache on device
        reinterpret_cast<int *>(position_ids.data_ptr<int>()), // <- pointer to index vector to gather cos_cache on device
        (int)(sparse_budget / chunk_size),                                         // gather A indices batch stride
        0,                                                     // gather B indices batch stride
        0,                                                     // scatter D indices batch stride
        (int)(sparse_budget / chunk_size),                                         // gather sin_cache indices batch stride
        (int)(sparse_budget / chunk_size),                                         // gather cost_cache indices batch stride
        (int)max_seq_len,
        (int)chunk_size,
        (int)heads,
        reinterpret_cast<int *>(offset_array.data_ptr<int>())
        }; // <- pointer to index vector to scatter D on device

    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size = Gemm::get_workspace_size(arguments);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Instantiate CUTLASS kernel depending on templates
    Gemm gemm_op;

    // Check the problem size is supported or not
    cutlass::Status status = gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);

    // Initialize CUTLASS kernel with arguments and workspace pointer
    status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);

    status = gemm_op();
    CUTLASS_CHECK(status);
}
