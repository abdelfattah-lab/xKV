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

# Build instructions:
# python setup.py build_ext --inplace
# Or to build a wheel:
# python setup.py bdist_wheel
#
# To specify a custom build directory:
# python setup.py build_ext --build-lib=/path/to/output/directory

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# efficiency/ is one level below repo root; cutlass lives at repo_root/3rdparty/cutlass
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root   = os.path.dirname(current_dir)

# Default build output directory (can be overridden with --build-lib)
# Set to 'build/lib' to keep artifacts out of the source directory
build_lib = os.path.join(current_dir, 'build', 'lib')

# Custom BuildExtension to set default build directory
class CustomBuildExtension(BuildExtension):
    def finalize_options(self):
        # Set build_lib if not already specified
        if self.build_lib is None:
            self.build_lib = build_lib
        super().finalize_options()

setup(
    name='xkv_sr',
    version='0.1.0',
    packages=find_packages(include=['ops', 'ops.*']),
    ext_modules=[
        CUDAExtension(
            name='ops._shadowkv',  # Place extension inside ops package
            sources=[
                os.path.join(current_dir, 'csrc/main.cu'),
                os.path.join(current_dir, 'csrc/rope.cu'),
                os.path.join(current_dir, 'csrc/rope_new.cu'),
                os.path.join(current_dir, 'csrc/gather_copy.cu'),
                os.path.join(current_dir, 'csrc/batch_gather_gemm.cu'),
                os.path.join(current_dir, 'csrc/batch_gemm_softmax.cu'),
            ],
            include_dirs=[
                os.path.join(repo_root, '3rdparty/cutlass/include'),
                os.path.join(repo_root, '3rdparty/cutlass/examples/common'),
                os.path.join(repo_root, '3rdparty/cutlass/tools/util/include'),
                os.path.join(current_dir, 'csrc'),
            ],
            extra_compile_args={
                'cxx': [
                    '-std=c++17',
                    '-O3',
                ],
                'nvcc': [
                    '-std=c++17',
                    '--expt-relaxed-constexpr',
                    '--expt-extended-lambda',
                    '-O3',
                    '-use_fast_math',
                    '-lcuda',
                    '-lcudart',
                ],
            },
        )
    ],
    cmdclass={
        'build_ext': CustomBuildExtension
    },
)
