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
#include <cute/arch/xe_copy_4B.hpp>

namespace cute
{
struct XE_2D_U16x1x16_LD_N {
  using BlockShape = Shape<_1, _16>;

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    __spirv_Subgroup2DBlockLoadINTEL(2, 16, 1, 1, baseoffset, width, height, pitch, coord, (void*)dst);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
#endif
  }
};

struct XE_2D_U16x2x16_LD_N {
  using BlockShape = Shape<_2, _16>;

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    __spirv_Subgroup2DBlockLoadINTEL(2, 16, 2, 1, baseoffset, width, height, pitch, coord, (void*)dst);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
#endif
  }
};

struct XE_2D_U16x4x16_LD_N {
  using BlockShape = Shape<_4, _16>;

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    __spirv_Subgroup2DBlockLoadINTEL(2, 16, 4, 1, baseoffset, width, height, pitch, coord, (void*)dst);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
#endif
  }
};

struct XE_2D_U16x8x16_LD_N {
  using BlockShape = Shape<_8, _16>;

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    __spirv_Subgroup2DBlockLoadINTEL(2, 16, 8, 1, baseoffset, width, height, pitch, coord, (void*)dst);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(SYCL_INTEL_TARGET)
      __spirv_Subgroup2DBlockPrefetchINTEL(2, 16, 8, 1, (const void*)baseoffset, width, height, pitch, coord);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
#endif
    }
  };
};

struct XE_2D_U16x16x16_LD_N {
  using BlockShape = Shape<_16, _16>;

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    __spirv_Subgroup2DBlockLoadINTEL(2, 16, 16, 1, baseoffset, width, height, pitch, coord, (void*)dst);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(SYCL_INTEL_TARGET)
      __spirv_Subgroup2DBlockPrefetchINTEL(2, 16, 16, 1, (const void*)baseoffset, width, height, pitch, coord);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
#endif
    }
  };
};

struct XE_2D_U16x32x16_LD_N {
  using BlockShape = Shape<_32, _16>;

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    __spirv_Subgroup2DBlockLoadINTEL(2, 16, 32, 1, baseoffset, width, height, pitch, coord, (void*)dst);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(SYCL_INTEL_TARGET)
      __spirv_Subgroup2DBlockPrefetchINTEL(2, 16, 32, 1, (const void*)baseoffset, width, height, pitch, coord);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
#endif
    }
  };
};

struct XE_2D_U16x1x32_LD_N {
  using BlockShape = Shape<_1, _32>;

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    __spirv_Subgroup2DBlockLoadINTEL(2, 16, 1, 2, baseoffset, width, height, pitch, coord, (void*)dst);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(SYCL_INTEL_TARGET)
      __spirv_Subgroup2DBlockPrefetchINTEL(2, 16, 1, 2, (const void*)baseoffset, width, height, pitch, coord);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
#endif
    }
  };
};

struct XE_2D_U16x2x32_LD_N {
  using BlockShape = Shape<_2, _32>;

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    __spirv_Subgroup2DBlockLoadINTEL(2, 16, 2, 2, baseoffset, width, height, pitch, coord, (void*)dst);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(SYCL_INTEL_TARGET)
      __spirv_Subgroup2DBlockPrefetchINTEL(2, 16, 2, 2, (const void*)baseoffset, width, height, pitch, coord);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
#endif
    }
  };
};

struct XE_2D_U16x4x32_LD_N {
  using BlockShape = Shape<_4, _32>;

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    __spirv_Subgroup2DBlockLoadINTEL(2, 16, 4, 2, baseoffset, width, height, pitch, coord, (void*)dst);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(SYCL_INTEL_TARGET)
      __spirv_Subgroup2DBlockPrefetchINTEL(2, 16, 4, 2, (const void*)baseoffset, width, height, pitch, coord);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
#endif
    }
  };
};

struct XE_2D_U16x8x32_LD_N {
  using BlockShape = Shape<_8, _32>;

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    __spirv_Subgroup2DBlockLoadINTEL(2, 16, 8, 2, baseoffset, width, height, pitch, coord, (void*)dst);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(SYCL_INTEL_TARGET)
      __spirv_Subgroup2DBlockPrefetchINTEL(2, 16, 8, 2, (const void*)baseoffset, width, height, pitch, coord);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
#endif
    }
  };
};

struct XE_2D_U16x16x32_LD_N {
  using BlockShape = Shape<_16, _32>;

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    __spirv_Subgroup2DBlockLoadINTEL(2, 16, 16, 2, baseoffset, width, height, pitch, coord, (void*)dst);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(SYCL_INTEL_TARGET)
      __spirv_Subgroup2DBlockPrefetchINTEL(2, 16, 16, 2, (const void*)baseoffset, width, height, pitch, coord);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
#endif
    }
  };
};

struct XE_2D_U16x32x32_LD_N {
  using BlockShape = Shape<_32, _32>;

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    __spirv_Subgroup2DBlockLoadINTEL(2, 16, 32, 2, baseoffset, width, height, pitch, coord, (void*)dst);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(SYCL_INTEL_TARGET)
      __spirv_Subgroup2DBlockPrefetchINTEL(2, 16, 32, 2, (const void*)baseoffset, width, height, pitch, coord);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
#endif
    }
  };
};

struct XE_2D_U16x16x16_LD_V {
  using BlockShape = Shape<_16, _16>;

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    __spirv_Subgroup2DBlockLoadTransformINTEL(2, 16, 16, 1, (const void*)baseoffset, width, height, pitch, coord, (void*)dst);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(SYCL_INTEL_TARGET)
      __spirv_Subgroup2DBlockPrefetchINTEL(2, 16, 16, 1, (const void*)baseoffset, width, height, pitch, coord);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
#endif
    }
  };
};

struct XE_2D_U16x32x16_LD_V {
  using BlockShape = Shape<_32, _16>;

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    __spirv_Subgroup2DBlockLoadTransformINTEL(2, 16, 32, 1, (const void*)baseoffset, width, height, pitch, coord, (void*)dst);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(SYCL_INTEL_TARGET)
      __spirv_Subgroup2DBlockPrefetchINTEL(2, 16, 32, 1, (const void*)baseoffset, width, height, pitch, coord);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
#endif
    }
  };
};

struct XE_2D_U16x16x32_LD_V {
  using BlockShape = Shape<_16, _32>;

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    __spirv_Subgroup2DBlockLoadTransformINTEL(2, 16, 16, 2, (const void*)baseoffset, width, height, pitch, coord, (void*)dst);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(SYCL_INTEL_TARGET)
      __spirv_Subgroup2DBlockPrefetchINTEL(2, 16, 16, 2, (const void*)baseoffset, width, height, pitch, coord);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
#endif
    }
  };
};

struct XE_2D_U16x32x32_LD_V {
  using BlockShape = Shape<_32, _32>;

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    __spirv_Subgroup2DBlockLoadTransformINTEL(2, 16, 32, 2, (const void*)baseoffset, width, height, pitch, coord, (void*)dst);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(SYCL_INTEL_TARGET)
       __spirv_Subgroup2DBlockPrefetchINTEL(2, 16, 16, 2, (const void*)baseoffset, width, height, pitch, coord);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
#endif
    }
  };
};

struct XE_2D_U16x16x8_LD_T {
  using BlockShape = Shape<_8, _16>;
  using inst_dtype = uint32_t;

  static constexpr bool is_transpose = true;

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 2, "Expected T to have size 4");
    *reinterpret_cast<intel::uint4 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_transpose_u32_k4(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
#endif
  }
};

struct XE_2D_U16x16x16_LD_T {
  using BlockShape = Shape<_16, _16>;
  using inst_dtype = uint32_t;

  static constexpr bool is_transpose = true;

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    __spirv_Subgroup2DBlockLoadTransposeINTEL(4, 8, 16, 1, (const void*)baseoffset, width, height, pitch, coord, (void*)dst);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
#endif
  }
};

struct XE_2D_U16x1x16_ST_N {
  using BlockShape = Shape<_1, _16>;

  template <class T>
  CUTE_HOST_DEVICE static void copy(void *baseoffset, int width, int height,
                                    int pitch, intel::coord_t coord,
                                    const T *src) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(SYCL_INTEL_TARGET)
    // static_assert(sizeof(T) == 2, "Expected T to have size 2");
    __spirv_Subgroup2DBlockStoreINTEL(2, 16, 1, 1, (void*)src, (const void*)baseoffset, width, height, pitch, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
#endif
  }
};

struct XE_2D_U16x2x16_ST_N {
  using BlockShape = Shape<_2, _16>;

  template <class T>
  CUTE_HOST_DEVICE static void copy(void *baseoffset, int width, int height,
                                    int pitch, intel::coord_t coord,
                                    const T *src) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(SYCL_INTEL_TARGET)
    // static_assert(sizeof(T) == 2, "Expected T to have size 2");
    __spirv_Subgroup2DBlockStoreINTEL(2, 16, 2, 1, (void*)src, (const void*)baseoffset, width, height, pitch, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
#endif
  }
};

struct XE_2D_U16x4x16_ST_N {
  using BlockShape = Shape<_4, _16>;

  template <class T>
  CUTE_HOST_DEVICE static void copy(void *baseoffset, int width, int height,
                                    int pitch, intel::coord_t coord,
                                    const T *src) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(SYCL_INTEL_TARGET)
    // static_assert(sizeof(T) == 2, "Expected T to have size 2");
    __spirv_Subgroup2DBlockStoreINTEL(2, 16, 4, 1, (void*)src, (const void*)baseoffset, width, height, pitch, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
#endif
  }
};

struct XE_2D_U16x8x16_ST_N {
  using BlockShape = Shape<_8, _16>;

  template <class T>
  CUTE_HOST_DEVICE static void copy(void *baseoffset, int width, int height,
                                    int pitch, intel::coord_t coord,
                                    const T *src) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(SYCL_INTEL_TARGET)
    // static_assert(sizeof(T) == 2, "Expected T to have size 2");
    __spirv_Subgroup2DBlockStoreINTEL(2, 16, 8, 1, (void*)src, (const void*)baseoffset, width, height, pitch, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
#endif
  }
};
} // end namespace cute
