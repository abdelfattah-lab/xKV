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

#pragma once
#include "cutlass/cutlass.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_types.h"
namespace cutlass {

template <typename T>
struct shuffle_odd_even_negate;

template <typename T, int N>
struct shuffle_odd_even_negate<Array<T, N>> {
  static_assert(N % 2 == 0 && "Only can shuffle 2K elements.");

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const & rhs) const {
    Array<T, N> result;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result[2 * i + 1] = rhs[2 * i];
      result[2 * i] = -rhs[2 * i + 1];
    }

    return result;
  }
};

} // cutlass