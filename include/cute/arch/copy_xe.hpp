/***************************************************************************************************
 * Copyright (c) 2024 - 2024 Codeplay Software Ltd. All rights reserved.
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

#include <cute/arch/copy.hpp>
#include <cute/config.hpp>
#include <cute/util/sycl_vec.hpp>

namespace cute
{
#if defined(CUTLASS_ENABLE_SYCL)
  #define ARCH_PVC_ACTIVATED 1
#endif

#ifdef __SYCL_DEVICE_ONLY__
#define SYCL_DEVICE_BUILTIN(x) SYCL_EXTERNAL extern "C" x
#else
#define SYCL_DEVICE_BUILTIN(x)  \
  inline x { assert(false); }
#endif

SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_write_flat_u32_m8k16v1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, int2_ coord, uint8 data));
SYCL_DEVICE_BUILTIN(ushort8 __builtin_IB_subgroup_block_read_flat_u16_m8k16v1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, int2_ coord));
SYCL_DEVICE_BUILTIN(uint8 __builtin_IB_subgroup_block_read_flat_u32_m8k16v1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, int2_ coord));
SYCL_DEVICE_BUILTIN(ushort64 __builtin_IB_subgroup_block_read_flat_u16_m32k16v2(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, int2_ coord));
SYCL_DEVICE_BUILTIN(ushort32 __builtin_IB_subgroup_block_read_flat_u16_m16k16v2(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, int2_ coord));
SYCL_DEVICE_BUILTIN(ushort16 intel_subgroup_block_read_u16_m8k16v2(
    __global void *baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, int2_ coord));
SYCL_DEVICE_BUILTIN(ushort32 __builtin_IB_subgroup_block_read_flat_u16_m32k16v1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, int2_ coord));
SYCL_DEVICE_BUILTIN(uint16 __builtin_IB_subgroup_block_read_flat_u32_m16k16v1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, int2_ coord));

#undef SYCL_DEVICE_BUILTIN

struct XE_2D_U16x8x16x1x1_LD_N
{
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, int2_ coord,
                                    T *dst) {
    #if defined(ARCH_PVC_ACTIVATED)
      static_assert(sizeof(T) == 2, "Expected T to have size 2");
      *(ushort8 *)dst = __builtin_IB_subgroup_block_read_flat_u16_m8k16v1(
          (long)baseoffset, width - 1, height - 1, pitch - 1, coord);
    #else
      CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware")
    #endif
  }
};

struct XE_2D_U32x8x16x1x1_LD_N
{
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, int2_ coord,
                                    T *dst) {
    #if defined(ARCH_PVC_ACTIVATED)
      static_assert(sizeof(T) == 4, "Expected T to have size 4");
      *(uint8 *)dst = __builtin_IB_subgroup_block_read_flat_u32_m8k16v1(
            (long)baseoffset, width - 1, height - 1, pitch - 1, coord); 
    #else
      CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware")
    #endif                                     
  }
};

struct XE_2D_U16x16x16x1x1_LD_N
{
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, int2_ coord,
                                    T *dst) {
    #if defined(ARCH_PVC_ACTIVATED)                                  
      static_assert(sizeof(T) == 2, "Expected T to have size 2");
      *(uint8 *)dst = __builtin_IB_subgroup_block_read_flat_u32_m8k16v1(
            (long)baseoffset, width - 1, height - 1, pitch - 1, coord);
    #else
      CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware")
    #endif                                      
  }
};

struct XE_2D_U16x8x16x4x2_LD_N
{
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, int2_ coord,
                                    T *dst) {
    #if defined(ARCH_PVC_ACTIVATED)
      static_assert(sizeof(T) == 2, "Expected T to have size 2");
      *(ushort64 *)dst = __builtin_IB_subgroup_block_read_flat_u16_m32k16v2(
          long(baseoffset), width - 1, height - 1, pitch - 1, coord);
    #else
      CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware")
    #endif
  }
};

struct XE_2D_U16x8x16x2x2_LD_N
{
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, int2_ coord,
                                    T *dst) {
    #if defined(ARCH_PVC_ACTIVATED)
      static_assert(sizeof(T) == 2, "Expected T to have size 2");
      *(ushort32*) dst = __builtin_IB_subgroup_block_read_flat_u16_m16k16v2(
          long(baseoffset), width - 1, height - 1, pitch - 1, coord);
    #else
      CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware")
    #endif
  }
};

struct XE_2D_U16x8x16x1x2_LD_N
{
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, int2_ coord,
                                    T *dst) {
    #if defined(ARCH_PVC_ACTIVATED)
      static_assert(sizeof(T) == 2, "Expected T to have size 2");
      ushort16 tmp = (intel_subgroup_block_read_u16_m8k16v2(
          (__global void *)baseoffset, width, height, pitch, coord));
      *(ushort16 *)dst = *reinterpret_cast<ushort16 *>(&tmp);
    #else
      CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware")
    #endif
  }
};

struct XE_2D_U16x8x16x4x1_LD_N
{
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, int2_ coord,
                                    T *dst) {
    #if defined(ARCH_PVC_ACTIVATED)
      static_assert(sizeof(T) == 2, "Expected T to have size 2");
        *(ushort32*) dst = __builtin_IB_subgroup_block_read_flat_u16_m32k16v1(
            long(baseoffset), width - 1, height - 1, pitch - 1, coord);
    #else
      CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware")
    #endif
  }
};

struct XE_2D_U32x8x16x2x1_LD_N
{
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, int2_ coord,
                                    T *dst) {
    #if defined(ARCH_PVC_ACTIVATED)
      static_assert(sizeof(T) == 4, "Expected T to have size 4");
      uint16 tmp = __builtin_IB_subgroup_block_read_flat_u32_m16k16v1(
          long(baseoffset), width - 1, height - 1, pitch - 1, coord);
      *(uint16 *)dst = *reinterpret_cast<uint16 *>(&tmp);
    #else
      CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware")
    #endif
  }
};

struct XE_2D_U16x16x16x2x1_LD_N
{
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, int2_ coord,
                                    T *dst) {
    #if defined(ARCH_PVC_ACTIVATED)
      static_assert(sizeof(T) == 2, "Expected T to have size 2");
      uint16 tmp = __builtin_IB_subgroup_block_read_flat_u32_m16k16v1(
          long(baseoffset), width - 1, height - 1, pitch - 1, coord);
      *(uint16 *)dst = *reinterpret_cast<uint16 *>(&tmp);
    #else
      CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware")
    #endif
  }
};

struct XE_2D_U32x8x16x1x1_ST_N
{
  template <class T>
  CUTE_HOST_DEVICE static void copy(void *baseoffset, int width, int height,
                                    int pitch, int2_ coord, const T *src) {
    #if defined(ARCH_PVC_ACTIVATED)
      static_assert(sizeof(T) == 4, "Expected T to have size 4");
      __builtin_IB_subgroup_block_write_flat_u32_m8k16v1(
          (long)baseoffset, width - 1, height - 1, pitch - 1, coord,
          *(uint8 *)src);
    #else
      CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware")
    #endif
  }
};

} // end namespace cute
