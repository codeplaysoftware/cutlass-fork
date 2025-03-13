/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Codeplay Software CORPORATION & AFFILIATES. All rights reserved.
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

#include "cute/atom/mma_atom.hpp"
#include "cute/atom/copy_atom.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/layout/layout.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"

namespace cutlass {
namespace gemm {
namespace device {
using namespace cute;

template<
  class OperatorClass, class ArchTag,
  class ElementA, class LayoutA,
  class ElementB, class LayoutB,
  class ElementC, class LayoutC,
  class ElementAccumulator>
struct DefaultGemmGroupConfiguration {
  static_assert(sizeof(ElementA) == 0, "No valid DefaultGemmGroupConfiguration configuration exists.");
};


// Intel XE MMA F32BF16
template <typename LayoutA, typename LayoutB, typename LayoutC>
struct DefaultGemmGroupConfiguration<
    arch::OpClassTensorOp, arch::IntelPVC,
    bfloat16_t, LayoutA,
    bfloat16_t, LayoutB,
    float, LayoutC,
    float>
{
  using TileShape = Shape<_256, _256, _32>;

  using TiledMma =
      TiledMMA<MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>,
               Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>,
               Tile<Layout<Shape<_8, _8, _4>, Stride<_1, _32, _8>>,
                    Layout<Shape<_16, _4, _4>, Stride<_1, _64, _16>>, _32>>;

  // A
  static constexpr int kAlignmentA = 32;
  using DefaultOperandA = cutlass::gemm::device::detail::DefaultGemm_TensorOpXe_OperandA<
    bfloat16_t, LayoutA, kAlignmentA, 32>;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;

  // B
  static constexpr int kAlignmentB = 32;
  using DefaultOperandB = cutlass::gemm::device::detail::DefaultGemm_TensorOpXe_OperandB<
    bfloat16_t, LayoutB, kAlignmentB, 32>;
  using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

  using EpilogueOp = epilogue::fusion::LinearCombination<float, float>;

  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<
    epilogue::IntelPVCEpilogue,
    EpilogueOp,
    TileShape,
    decltype(tile_shape(TiledMma()))
  >;

  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
    epilogue::IntelPVCGroupEpilogue,
    TileShape,
    float, TagToStrideC_t<LayoutC*>,
    float, TagToStrideC_t<LayoutC*>,
    FusionCallBacks,
    XE_2D_U32x8x16_LD_N, void, void,
    XE_2D_U32x8x16_ST_N, void, void>;

  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
    MainloopIntelPVCGroup<2>, TileShape,
    bfloat16_t, cutlass::detail::TagToStrideA_t<LayoutA*>,
    bfloat16_t, cutlass::detail::TagToStrideB_t<LayoutB*>,
    TiledMma,
    GmemTiledCopyA, void, void, cute::identity,  // A
    GmemTiledCopyB, void, void, cute::identity   // B
  >;
};

// Intel XE MMA F32F16
template <typename LayoutA, typename LayoutB, typename LayoutC>
struct DefaultGemmGroupConfiguration<
    arch::OpClassTensorOp, arch::IntelPVC,
    half_t, LayoutA,
    half_t, LayoutB,
    float, LayoutC,
    float>
{
  using TileShape = Shape<_256, _256, _32>;

  using TiledMma =
      TiledMMA<MMA_Atom<XE_8x16x16_F32F16F16F32_TT>,
               Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>,
               Tile<Layout<Shape<_8, _8, _4>, Stride<_1, _32, _8>>,
                    Layout<Shape<_16, _4, _4>, Stride<_1, _64, _16>>, _32>>;

  // A
  static constexpr int kAlignmentA = 32;
  using DefaultOperandA = cutlass::gemm::device::detail::DefaultGemm_TensorOpXe_OperandA<
    half_t, LayoutA, kAlignmentA, 32>;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;

  // B
  static constexpr int kAlignmentB = 32;
  using DefaultOperandB = cutlass::gemm::device::detail::DefaultGemm_TensorOpXe_OperandB<
    half_t, LayoutB, kAlignmentB, 32>;
  using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

  using EpilogueOp = epilogue::fusion::LinearCombination<float, float>;

  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<
    epilogue::IntelPVCEpilogue,
    EpilogueOp,
    TileShape,
    decltype(tile_shape(TiledMma()))
  >;

  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
    epilogue::IntelPVCGroupEpilogue,
    TileShape,
    float, TagToStrideC_t<LayoutC*>,
    float, TagToStrideC_t<LayoutC*>,
    FusionCallBacks,
    XE_2D_U32x8x16_LD_N, void, void,
    XE_2D_U32x8x16_ST_N, void, void>;

  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
    MainloopIntelPVCGroup<2>, TileShape,
    half_t, cutlass::detail::TagToStrideA_t<LayoutA*>,
    half_t, cutlass::detail::TagToStrideB_t<LayoutB*>,
    TiledMma,
    GmemTiledCopyA, void, void, cute::identity,  // A
    GmemTiledCopyB, void, void, cute::identity   // B
  >;
};


} // namespace device
} // namespace gemm
} // namespace cutlass
