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

#include "cute/arch/mma_sm80.hpp"
#include "cute/atom/mma_traits_sm80.hpp"
#include "cute/arch/copy_sm80.hpp"


namespace cutlass::gemm::collective::detail {
  //================== MMA Types ==================//

  template<typename ElementA, typename ElementB, typename ElementAccum>
  struct Sm80_TiledMMA {
       using MMA_Atom = MMA_Atom<UniversalFMA<ElementAccum, ElementA, ElementB, ElementAccum>>;
       using TiledMMA = TiledMMA<MMA_Atom, Layout<Shape<_4, _4, _1>>>;
  };

  template<>
  struct Sm80_TiledMMA<cute::half_t, cute::half_t, cute::half_t> {
       using MMA_Atom = MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>;
       using TiledMMA = TiledMMA<MMA_Atom,
                                 Layout<Shape<_2, _2, _1>>,
                                 Tile<_32, _32, _16>>;
  };

  template<>
  struct Sm80_TiledMMA<cute::half_t, cute::half_t, float> {
       using MMA_Atom = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
       using TiledMMA = TiledMMA<MMA_Atom,
                                 Layout<Shape<_2, _2, _1>>,
                                 Tile<_32, _32, _16>>;
  };

  template<>
  struct Sm80_TiledMMA<cute::bfloat16_t, cute::bfloat16_t, float> {
       using MMA_Atom = MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>;
       using TiledMMA = TiledMMA<MMA_Atom, 
                                 Layout<Shape<_2, _2, _1>>,
                                 Tile<_32, _32, _16>>;
  };

  template<>
  struct Sm80_TiledMMA<cute::tfloat32_t, cute::tfloat32_t, float> {
       using MMA_Atom = MMA_Atom<SM80_16x8x8_F32TF32TF32F32_TN>;
       using TiledMMA = TiledMMA<MMA_Atom,
                                 Layout<Shape<_2, _2, _1>, Stride<_2, _1, _1>>,
                                 Tile<_32, _32, _8>>;
  };

  template<>
  struct Sm80_TiledMMA<double, double, double> {
       using MMA_Atom = MMA_Atom<SM80_8x8x4_F64F64F64F64_TN>;
       using TiledMMA = TiledMMA<MMA_Atom,
                                 Layout<Shape<_2, _2, _1>>,
                                 Tile<Layout<Shape<_16, _2>, Stride<_2, _1>>,
                                      Layout<Shape<_16, _2>, Stride<_2, _1>>,
                                      Underscore>>;
  };

  template<>
  struct Sm80_TiledMMA<int8_t, int8_t, int32_t> {
       using MMA_Atom = MMA_Atom<SM80_16x8x32_S32S8S8S32_TN_SATURATE>;
       using TiledMMA = TiledMMA<MMA_Atom,
                                 Layout<Shape<_2, _2, _1>>,
                                 Tile<_32, _32, _32>>;
  };

  template<>
  struct Sm80_TiledMMA<uint8_t, uint8_t, int32_t> {
       using MMA_Atom = MMA_Atom<SM80_16x8x32_S32U8U8S32_TN_SATURATE>;
       using TiledMMA = TiledMMA<MMA_Atom,
                                 Layout<Shape<_2, _2, _1>>,
                                 Tile<_32, _32, _32>>;
  };

  template<>
  struct Sm80_TiledMMA<int8_t, uint8_t, int32_t> {
       using MMA_Atom = MMA_Atom<SM80_16x8x32_S32S8U8S32_TN_SATURATE>;
       using TiledMMA = TiledMMA<MMA_Atom,
                                 Layout<Shape<_2, _2, _1>>,
                                 Tile<_32, _32, _32>>;
  };

  template<>
  struct Sm80_TiledMMA<uint8_t, int8_t, int32_t> {
       using MMA_Atom = MMA_Atom<SM80_16x8x32_S32U8S8S32_TN_SATURATE>;
       using TiledMMA = TiledMMA<MMA_Atom,
                                 Layout<Shape<_2, _2, _1>>,
                                 Tile<_32, _32, _32>>;
  };

  //////////////////////////////////////////////////////////////////////////////////////////////////

  template<typename LayoutA, typename ElementA>
  struct Sm80_MemoryAtomsA;

  template<typename LayoutB, typename ElementB> 
  struct Sm80_MemoryAtomsB;

  template<>
  struct Sm80_MemoryAtomsA<cutlass::layout::RowMajor, cute::half_t> {
       using SmemLayoutAtom = decltype(
              composition(Swizzle<3, 3, 3>{},
                          Layout<Shape<_8, _64>,
                                 Stride<_64, _1>>{}));
       using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, cute::half_t>;

       using GmemTiledCopy = decltype(
              make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, cute::half_t>{},
                          Layout<Shape<_16, _8>,
                                 Stride<_8, _1>>{},
                          Layout<Shape<_1, _8>>{}));
  };

  template<>
  struct Sm80_MemoryAtomsA<cutlass::layout::ColumnMajor, cute::half_t> {
       using SmemLayoutAtom = decltype(
              composition(Swizzle<3, 3, 3>{},
                          Layout<Shape<_64, _8>,
                                 Stride<_1, _64>>{}));
       using SmemCopyAtom = Copy_Atom<SM75_U16x8_LDSM_T, cute::half_t>;

       using GmemTiledCopy = decltype(
              make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, cute::half_t>{},
                          Layout<Shape<_16, _8>,
                                 Stride<_1, _16>>{},
                          Layout<Shape<_8, _1>>{}));
  };

  template<>
  struct Sm80_MemoryAtomsB<cutlass::layout::RowMajor, cute::half_t> :
         Sm80_MemoryAtomsA<cutlass::layout::ColumnMajor, cute::half_t>{};

  template<>
  struct Sm80_MemoryAtomsB<cutlass::layout::ColumnMajor, cute::half_t> : 
         Sm80_MemoryAtomsA<cutlass::layout::RowMajor, cute::half_t>{};

  // We can re-use half_t memory layouts for bf16 as well
  template<>
  struct Sm80_MemoryAtomsA<cutlass::layout::RowMajor, cute::bfloat16_t> :
         Sm80_MemoryAtomsA<cutlass::layout::RowMajor, cute::half_t>{};

  template<>
  struct Sm80_MemoryAtomsA<cutlass::layout::ColumnMajor, cute::bfloat16_t> : 
         Sm80_MemoryAtomsA<cutlass::layout::ColumnMajor, cute::half_t>{};

  template<>
  struct Sm80_MemoryAtomsB<cutlass::layout::RowMajor, cute::bfloat16_t> : 
         Sm80_MemoryAtomsB<cutlass::layout::RowMajor, cute::half_t>{};

  template<>
  struct Sm80_MemoryAtomsB<cutlass::layout::ColumnMajor, cute::bfloat16_t> : 
         Sm80_MemoryAtomsB<cutlass::layout::ColumnMajor, cute::half_t>{};

  template<>
  struct Sm80_MemoryAtomsA<cutlass::layout::RowMajor, cute::tfloat32_t> {
       using SmemLayoutAtom = decltype(
              composition(Swizzle<3, 3, 3>{},
                          Layout<Shape<_8, _32>,
                                 Stride<_32, _1>>{}));
       using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, cute::tfloat32_t>;

       using GmemTiledCopy = decltype(
              make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, tfloat32_t>{},
                              Layout<Shape<_16, _8>,
                                     Stride<_8, _1>>{},
                              Layout<Shape<_1, _4>>{}));
  };

  template<>
  struct Sm80_MemoryAtomsA<cutlass::layout::ColumnMajor, cute::tfloat32_t> {
       using SmemLayoutAtom = decltype(
              composition(Swizzle<3, 2, 3>{},
                          Layout<Shape<_32, _8>,
                                 Stride<_1, _32>>{}));
       using SmemCopyAtom = Copy_Atom<UniversalCopy<tfloat32_t>, tfloat32_t>;

       using GmemTiledCopy = decltype(
              make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, tfloat32_t>{},
                              Layout<Shape<_16, _8>,
                                     Stride<_1, _16>>{},
                              Layout<Shape<_4, _1>>{}));
  };


  template<>
  struct Sm80_MemoryAtomsB<cutlass::layout::RowMajor, cute::tfloat32_t> : 
         Sm80_MemoryAtomsA<cutlass::layout::ColumnMajor, cute::tfloat32_t>{};

  template<>
  struct Sm80_MemoryAtomsB<cutlass::layout::ColumnMajor, cute::tfloat32_t> :
         Sm80_MemoryAtomsA<cutlass::layout::RowMajor, cute::tfloat32_t>{};


  template<>
  struct Sm80_MemoryAtomsA<cutlass::layout::RowMajor, double> {
       using SmemLayoutAtom = decltype(
              composition(Swizzle<2, 0, 4>{},
                          Layout<Shape<_4, _16>,
                                 Stride<_1, _4>>{}));
       using SmemCopyAtom = Copy_Atom<DefaultCopy, double>;

       using GmemTiledCopy = decltype(
              make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<double>, double>{},
                              Layout<Shape<_8, _16>,
                                     Stride<_16, _1>>{},
                              Layout<Shape<_1, _1>>{}));
  };

  template<>
  struct Sm80_MemoryAtomsA<cutlass::layout::ColumnMajor, double> {
       using SmemLayoutAtom = decltype(
              composition(Swizzle<2, 2, 2>{},
                          Layout<Shape<_16, _4>,
                                 Stride<_1, _16>>{}));
       using SmemCopyAtom = Copy_Atom<DefaultCopy, double>;

       using GmemTiledCopy = decltype(
              make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, double>{},
                              Layout<Shape<_16, _8>,
                                     Stride<_1, _16>>{},
                              Layout<Shape<_2, _1>>{}));
  };

  template<>
  struct Sm80_MemoryAtomsB<cutlass::layout::RowMajor, double> : 
         Sm80_MemoryAtomsA<cutlass::layout::ColumnMajor, double>{};

  template<>
  struct Sm80_MemoryAtomsB<cutlass::layout::ColumnMajor, double> : 
         Sm80_MemoryAtomsA<cutlass::layout::RowMajor, double>{};

  template<>
  struct Sm80_MemoryAtomsA<cutlass::layout::RowMajor, int8_t> {
       using SmemLayoutAtom = decltype(
              composition(Swizzle<2, 4, 3>{},
              Layout<Shape<_16, _64>,
                      Stride<_64, _1>>{}));
       using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, int8_t>;

       using GmemTiledCopy = decltype(
              make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, int8_t>{},
                              Layout<Shape<_32, _4>,
                                     Stride<_4, _1>>{},
                               Layout<Shape<_1, _16>>{}));
  };

  template<>
  struct Sm80_MemoryAtomsA<cutlass::layout::ColumnMajor, int8_t> {
       using SmemLayoutAtom = decltype(
              composition(Swizzle<2, 0, 8>{},
                          Layout<Shape<_64, _16>,
                                 Stride<_1, _64>>{}));
       using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, int8_t>;

       using GmemTiledCopy = decltype(
      make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, int8_t>{},
                      Layout<Shape <_4, _32>,
                            Stride<_1, _4>>{},
                      Layout<Shape<_1,Int<16>>>{}));
  };

  template<>
  struct Sm80_MemoryAtomsB<cutlass::layout::ColumnMajor, int8_t> : 
         Sm80_MemoryAtomsA<cutlass::layout::RowMajor, int8_t>{};

  template<>
  struct Sm80_MemoryAtomsB<cutlass::layout::RowMajor, int8_t> : 
         Sm80_MemoryAtomsA<cutlass::layout::ColumnMajor, int8_t>{};

  template<>
  struct Sm80_MemoryAtomsA<cutlass::layout::RowMajor, uint8_t> {
       using SmemLayoutAtom = decltype(
              composition(Swizzle<2, 4, 3>{},
              Layout<Shape<_16, _64>,
                      Stride<_64, _1>>{}));
       using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, uint8_t>;

       using GmemTiledCopy = decltype(
              make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, uint8_t>{},
                              Layout<Shape<_32, _4>,
                                     Stride<_4, _1>>{},
                               Layout<Shape<_1, _16>>{}));
  };

  template<>
  struct Sm80_MemoryAtomsA<cutlass::layout::ColumnMajor, uint8_t> {
       using SmemLayoutAtom = decltype(
              composition(Swizzle<2, 0, 8>{},
                          Layout<Shape<_64, _16>,
                                 Stride<_1, _64>>{}));
       using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, uint8_t>;

       using GmemTiledCopy = decltype(
      make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, uint8_t>{},
                      Layout<Shape <_4, _32>,
                            Stride<_1, _4>>{},
                      Layout<Shape<_1,Int<16>>>{}));
  };

  template<>
  struct Sm80_MemoryAtomsB<cutlass::layout::ColumnMajor, uint8_t> : 
         Sm80_MemoryAtomsA<cutlass::layout::RowMajor, uint8_t>{};

  template<>
  struct Sm80_MemoryAtomsB<cutlass::layout::RowMajor, uint8_t> : 
         Sm80_MemoryAtomsA<cutlass::layout::ColumnMajor, uint8_t>{};
}
