/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

/**
 * \file
 * \brief Debugging and logging functionality
 */

#include <cuda_runtime_api.h>

#include <cute/config.hpp>

#ifdef CUTLASS_ENABLE_SYCL
#include <syclcompat.hpp>
#endif

namespace cute
{

/******************************************************************************
 * Debug and logging macros
 ******************************************************************************/

/**
 * Formats and prints the given message to stdout
 */
#if !defined(CUTE_LOG)
#  if !defined(__CUDA_ARCH__)
#    define CUTE_LOG(format, ...) printf(format, __VA_ARGS__)
#  else
#    define CUTE_LOG(format, ...)                                \
        printf("[block (%d,%d,%d), thread (%d,%d,%d)]: " format, \
               blockIdx.x,  blockIdx.y,  blockIdx.z,             \
               threadIdx.x, threadIdx.y, threadIdx.z,            \
               __VA_ARGS__);
#  endif
#endif

/**
 * Formats and prints the given message to stdout only if DEBUG is defined
 */
#if !defined(CUTE_LOG_DEBUG)
#  ifdef DEBUG
#    define CUTE_LOG_DEBUG(format, ...) CUTE_LOG(format, __VA_ARGS__)
#  else
#    define CUTE_LOG_DEBUG(format, ...)
#  endif
#endif

/**
 * \brief Perror macro with exit
 */
#if !defined(CUTE_ERROR_EXIT)
#  define CUTE_ERROR_EXIT(e)                                         \
      do {                                                           \
        cudaError_t code = (e);                                      \
        if (code != cudaSuccess) {                                   \
          fprintf(stderr, "<%s:%d> %s:\n    %s: %s\n",               \
                  __FILE__, __LINE__, #e,                            \
                  cudaGetErrorName(code), cudaGetErrorString(code)); \
          fflush(stderr);                                            \
          exit(1);                                                   \
        }                                                            \
      } while (0)
#endif

#if !defined(CUTE_CHECK_LAST)
#  define CUTE_CHECK_LAST() CUTE_ERROR_EXIT(cudaPeekAtLastError()); CUTE_ERROR_EXIT(cudaDeviceSynchronize())
#endif

#if !defined(CUTE_CHECK_ERROR)
#  define CUTE_CHECK_ERROR(e) CUTE_ERROR_EXIT(e)
#endif

// A dummy function that uses compilation failure to print a type
template <class... T>
CUTE_HOST_DEVICE void
print_type() {
  static_assert(sizeof...(T) < 0, "Printing type T.");
}

template <class... T>
CUTE_HOST_DEVICE void
print_type(T&&...) {
  static_assert(sizeof...(T) < 0, "Printing type T.");
}

//
// Device-specific helpers
//
// e.g.
// if (thread0()) print(...);
// if (block0()) print(...);
// if (thread(42)) print(...);

CUTE_HOST_DEVICE
bool
block(int bid)
{
#if defined(CUTLASS_ENABLE_SYCL)
    using sycl::ext::oneapi::experimental::this_nd_item;
    return (this_nd_item<3>.get_linear_id()==bid);
#elif defined(__CUDA_ARCH__)
  return blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y == bid;
#elif defined(CUTLASS_ENABLE_SYCL)
    using namespace syclcompat;
    return (work_group_id::x() +
       work_group_id::y() * global_range::x() +
       work_group_id::z() * global_range::x() * global_range::y() == bid);
#else
  return true;
#endif
}

CUTE_HOST_DEVICE
bool
thread(int tid, int bid)
{
#if defined(CUTLASS_ENABLE_SYCL)
    using namespace syclcompat;
    return (local_id::x() +
       local_id::y() * work_group_range::x() +
       local_id::z() * work_group_range::x() * work_group_range::y() == tid) && block(bid);
#elif defined(__CUDA_ARCH__)
  return (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y == tid) && block(bid);
#elif defined(CUTLASS_ENABLE_SYCL)
    using namespace syclcompat;
    return (local_id::x() +
       local_id::y() * work_group_range::x() +
       local_id::z() * work_group_range::x() * work_group_range::y() == tid) && block(bid);
#else
  return true;
#endif
}

CUTE_HOST_DEVICE
bool
thread(int tid)
{
  return thread(tid,0);
}

CUTE_HOST_DEVICE
bool
thread0()
{
  return thread(0,0);
}

CUTE_HOST_DEVICE
bool
block0()
{
  return block(0);
}

}  // end namespace cute
