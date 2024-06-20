# Copyright (c) 2024 - 2024 Codeplay Software Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

include_guard()

include(ExternalProject)

set(ONEMKL_INSTALL_DIR ${CMAKE_BINARY_DIR}/deps/oneMKL)
set(ONEMKL_INCLUDE_DIR ${ONEMKL_INSTALL_DIR}/include)
set(ONEMKL_LIB_DIR ${ONEMKL_INSTALL_DIR}/lib)
set(ONEMKL_LIB ${ONEMKL_LIB_DIR}/libonemkl.so)

if (NOT DEFINED MKL_ROOT)
  set(MKL_ROOT "/opt/intel/oneapi/mkl/latest")
endif()

if(SYCL_NVIDIA_TARGET)
  set(ENABLE_CUBLAS_BACKEND ON)
  set(ENABLE_MKLGPU_BACKEND OFF)
elseif(SYCL_INTEL_TARGET)
  set(ENABLE_CUBLAS_BACKEND OFF)
  set(ENABLE_MKLGPU_BACKEND ON)
else()
  message(FATAL_ERROR "SYCL target required for MKL compilation")
endif()
ExternalProject_Add(
    onemkl_project

    PREFIX                  ${ONEMKL_INSTALL_DIR}
    GIT_REPOSITORY          "https://github.com/oneapi-src/oneMKL.git"
    GIT_TAG                 "v0.4"

    CMAKE_ARGS
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_GENERATOR=${CMAKE_GENERATOR}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX=${ONEMKL_INSTALL_DIR}
    -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath=${ONEMKL_LIB_DIR}"
    -DTARGET_DOMAINS=blas
    -DENABLE_MKLCPU_BACKEND=OFF
    -DENABLE_MKLGPU_BACKEND=${ENABLE_MKLGPU_BACKEND}
    -DENABLE_CUBLAS_BACKEND=${ENABLE_CUBLAS_BACKEND}
    -DBUILD_FUNCTIONAL_TESTS=OFF
    -DBUILD_EXAMPLES=OFF
    -DBUILD_DOC=OFF
    -DMKL_ROOT=${MKL_ROOT}
    INSTALL_DIR ${ONEMKL_INSTALL_DIR}
    BUILD_BYPRODUCTS ${ONEMKL_LIB}
)

add_library(oneMKL SHARED IMPORTED)
include_directories(${ONEMKL_INSTALL_DIR}/include)
