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

#include <cute/arch/mma_xe.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits.hpp>

#include <cute/layout.hpp>

namespace cute
{
template <class... Args>
struct MMA_Atom;

template <>
struct MMA_Traits<XE_8x16x16_F32BF16BF16F32_TT>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_8,_16,_16>;
  using ThrID   = Layout<_16>;

  using ALayout = Layout<Shape<_16, _8>, Stride<_8, _1>>;
  using BLayout = Layout<Shape<_16, _16>, Stride<_1, _16>>;
  using CLayout = Layout<Shape<_16, _8>, Stride<_8, _1>>;
};

template <>
struct MMA_Traits<XE_4x16x16_F32BF16BF16F32_TT>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_4,_16,_16>;
  using ThrID   = Layout<_16>;

  using ALayout = Layout<Shape<_16, _4>, Stride<_4, _1>>;
  using BLayout = Layout<Shape<_16, _16>, Stride<_1, _16>>;
  using CLayout = Layout<Shape<_16, _4>, Stride<_4, _1>>;
};

template <>
struct MMA_Traits<XE_2x16x16_F32BF16BF16F32_TT>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_2,_16,_16>;
  using ThrID   = Layout<_16>;

  using ALayout = Layout<Shape<_16, _2>, Stride<_2, _1>>;
  using BLayout = Layout<Shape<_16, _16>, Stride<_1, _16>>;
  using CLayout = Layout<Shape<_16, _2>, Stride<_2, _1>>;
};

template <>
struct MMA_Traits<XE_1x16x16_F32BF16BF16F32_TT>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_1,_16,_16>;
  using ThrID   = Layout<_16>;

  using ALayout = Layout<Shape<_16, _1>, Stride<_1, _1>>;
  using BLayout = Layout<Shape<_16, _16>, Stride<_1, _16>>;
  using CLayout = Layout<Shape<_16, _1>, Stride<_1, _1>>;
};


template <>
struct MMA_Traits<XE_8x16x16_F32F16F16F32_TT>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_8,_16,_16>;
  using ThrID   = Layout<_16>;
  using ALayout = Layout<Shape<_16, _8>, Stride<_8, _1>>;
  using BLayout = Layout<Shape<_16, _16>, Stride<_1, _16>>;
  using CLayout = Layout<Shape<_16, _8>, Stride<_8, _1>>;
};

template <>
struct MMA_Traits<XE_4x16x16_F32F16F16F32_TT>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_4,_16,_16>;
  using ThrID   = Layout<_16>;
  using ALayout = Layout<Shape<_16, _4>, Stride<_4, _1>>;
  using BLayout = Layout<Shape<_16, _16>, Stride<_1, _16>>;
  using CLayout = Layout<Shape<_16, _4>, Stride<_4, _1>>;
};

template <>
struct MMA_Traits<XE_2x16x16_F32F16F16F32_TT>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_2,_16,_16>;
  using ThrID   = Layout<_16>;
  using ALayout = Layout<Shape<_16, _2>, Stride<_2, _1>>;
  using BLayout = Layout<Shape<_16, _16>, Stride<_1, _16>>;
  using CLayout = Layout<Shape<_16, _2>, Stride<_2, _1>>;
};

template <>
struct MMA_Traits<XE_1x16x16_F32F16F16F32_TT>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_1,_16,_16>;
  using ThrID   = Layout<_16>;
  using ALayout = Layout<Shape<_16, _1>, Stride<_1, _1>>;
  using BLayout = Layout<Shape<_16, _16>, Stride<_1, _16>>;
  using CLayout = Layout<Shape<_16, _1>, Stride<_1, _1>>;
};

template <>
struct MMA_Traits<XE_8x16x32_S32S8S8S32_TT>
{
  using ValTypeD = int;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int;

  using Shape_MNK = Shape<_8,_16,_32>;
  using ThrID   = Layout<_16>;
  using ALayout = Layout<Shape<_16, Shape<_2, _8>>, Stride<_16, Stride<_8, _1>>>;
  using BLayout = Layout<Shape<_16, _32>, Stride<_1, _16>>;
  using CLayout = Layout<Shape<_16, _8>, Stride<_8, _1>>;
};

template <>
struct MMA_Traits<XE_4x16x32_S32S8S8S32_TT>
{
  using ValTypeD = int;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int;

  using Shape_MNK = Shape<_4,_16,_32>;
  using ThrID   = Layout<_16>;
  using ALayout = Layout<Shape<_16, Shape<_2, _4>>, Stride<_8, Stride<_4, _1>>>;
  using BLayout = Layout<Shape<_16, _32>, Stride<_1, _16>>;
  using CLayout = Layout<Shape<_16, _4>, Stride<_4, _1>>;
};

template <>
struct MMA_Traits<XE_2x16x32_S32S8S8S32_TT>
{
  using ValTypeD = int;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int;

  using Shape_MNK = Shape<_2,_16,_32>;
  using ThrID   = Layout<_16>;
  using ALayout = Layout<Shape<_16, Shape<_2, _2>>, Stride<_4, Stride<_2, _1>>>;
  using BLayout = Layout<Shape<_16, _32>, Stride<_1, _16>>;
  using CLayout = Layout<Shape<_16, _2>, Stride<_2, _1>>;
};

template <>
struct MMA_Traits<XE_1x16x32_S32S8S8S32_TT>
{
  using ValTypeD = int;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int;

  using Shape_MNK = Shape<_1,_16,_32>;
  using ThrID   = Layout<_16>;
  using ALayout = Layout<Shape<_16, Shape<_2, _1>>, Stride<_2, Stride<_1, _1>>>;
  using BLayout = Layout<Shape<_16, _32>, Stride<_1, _16>>;
  using CLayout = Layout<Shape<_16, _1>, Stride<_1, _1>>;
};

template <>
struct MMA_Traits<XE_8x16x32_S32U8U8S32_TT>
{
  using ValTypeD = int;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int;

  using Shape_MNK = Shape<_8,_16,_32>;
  using ThrID   = Layout<_16>;
  using ALayout = Layout<Shape<_16, Shape<_2, _8>>, Stride<_16, Stride<_8, _1>>>;
  using BLayout = Layout<Shape<_16, _32>, Stride<_1, _16>>;
  using CLayout = Layout<Shape<_16, _8>, Stride<_8, _1>>;
};

template <>
struct MMA_Traits<XE_4x16x32_S32U8U8S32_TT>
{
  using ValTypeD = int;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int;

  using Shape_MNK = Shape<_4,_16,_32>;
  using ThrID   = Layout<_16>;
  using ALayout = Layout<Shape<_16, Shape<_2, _4>>, Stride<_8, Stride<_4, _1>>>;
  using BLayout = Layout<Shape<_16, _32>, Stride<_1, _16>>;
  using CLayout = Layout<Shape<_16, _4>, Stride<_4, _1>>;
};

template <>
struct MMA_Traits<XE_2x16x32_S32U8U8S32_TT>
{
  using ValTypeD = int;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int;

  using Shape_MNK = Shape<_2,_16,_32>;
  using ThrID   = Layout<_16>;
  using ALayout = Layout<Shape<_16, Shape<_2, _2>>, Stride<_4, Stride<_2, _1>>>;
  using BLayout = Layout<Shape<_16, _32>, Stride<_1, _16>>;
  using CLayout = Layout<Shape<_16, _2>, Stride<_2, _1>>;
};

template <>
struct MMA_Traits<XE_1x16x32_S32U8U8S32_TT>
{
  using ValTypeD = int;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int;

  using Shape_MNK = Shape<_1,_16,_32>;
  using ThrID   = Layout<_16>;
  using ALayout = Layout<Shape<_16, Shape<_2, _1>>, Stride<_2, Stride<_1, _1>>>;
  using BLayout = Layout<Shape<_16, _32>, Stride<_1, _16>>;
  using CLayout = Layout<Shape<_16, _1>, Stride<_1, _1>>;
};

template <>
struct MMA_Traits<XE_8x16x8_F32TF32TF32F32_TT>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_8,_16,_8>;
  using ThrID   = Layout<_16>;
  using ALayout = Layout<Shape<Shape<_8, _2>, _4>, Stride<Stride<_8, _1>, _2>>;
  using BLayout = Layout<Shape<_16, _8>, Stride<_1, _16>>;
  using CLayout = Layout<Shape<_16, _8>, Stride<_8, _1>>;
};

template <>
struct MMA_Traits<XE_4x16x8_F32TF32TF32F32_TT>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_4,_16,_8>;
  using ThrID   = Layout<_16>;
  using ALayout = Layout<Shape<Shape<_8, _2>, _2>, Stride<Stride<_4, _1>, _2>>;
  using BLayout = Layout<Shape<_16, _8>, Stride<_1, _16>>;
  using CLayout = Layout<Shape<_16, _4>, Stride<_4, _1>>;
};

template <>
struct MMA_Traits<XE_2x16x8_F32TF32TF32F32_TT>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_2,_16,_8>;
  using ThrID   = Layout<_16>;
  using ALayout = Layout<Shape<Shape<_8, _2>, _1>, Stride<Stride<_2, _1>, _0>>;
  using BLayout = Layout<Shape<_16, _8>, Stride<_1, _16>>;
  using CLayout = Layout<Shape<_16, _2>, Stride<_2, _1>>;
};

template <>
struct MMA_Traits<XE_1x16x8_F32TF32TF32F32_TT>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_1,_16,_8>;
  using ThrID   = Layout<_16>;
  using ALayout = Layout<Shape<Shape<_8, _2>, _1>, Stride<Stride<_1, _0>, _0>>;
  using BLayout = Layout<Shape<_16, _8>, Stride<_1, _16>>;
  using CLayout = Layout<Shape<_16, _1>, Stride<_1, _1>>;
};

template <class MMA, class ThrCoord>
struct ThrMMA;

template <class XeMMA,
          class ThrVMNK>
struct XeThrMMA  {
  ThrMMA<XeMMA, ThrVMNK> thr_mma;

  CUTE_HOST_DEVICE
  XeThrMMA(XeMMA const& mma, ThrVMNK const& vmnk) {
    thr_mma = ThrMMA<XeMMA, ThrVMNK>{mma, vmnk};
  }

  template <class ATensor>
  CUTE_HOST_DEVICE constexpr
  auto
  partition_fragment_A(ATensor&& atensor) const
  {
    return thr_mma.partition_fragment_A(atensor);
  }

  template <class BTensor>
  CUTE_HOST_DEVICE constexpr
  auto
  partition_fragment_B(BTensor&& btensor) const
  {
    return  thr_mma.partition_fragment_B(btensor);
  }

};

template <class MMA_Atom,
          class AtomLayoutMNK,
          class PermutationMNK>
struct TiledMMA;

template <class MMA_Atom,
          class AtomLayoutMNK,
          class PermutationMNK = Tile<Underscore,Underscore,Underscore>>
struct XeTiledMMA : TiledMMA<MMA_Atom, AtomLayoutMNK, PermutationMNK> {

  CUTE_HOST_DEVICE constexpr
  XeTiledMMA(MMA_Atom const& mma_atom = {}, AtomLayoutMNK const& thr_layout_mnk = {})
            : TiledMMA<MMA_Atom, AtomLayoutMNK, PermutationMNK>(mma_atom, thr_layout_mnk)
  {}

  template <class ThrIdx,
            __CUTE_REQUIRES(is_integral<ThrIdx>::value)>
  CUTE_HOST_DEVICE constexpr
  auto
  get_slice(ThrIdx const& thr_idx) const
  {
    auto thr_vmnk = this->thr_layout_vmnk_.get_flat_coord(thr_idx);
    return XeThrMMA<XeTiledMMA, decltype(thr_vmnk)>{*this, thr_vmnk};
  }

};

template <class MMA_Op,
          class MMAThrLayout = Layout<Shape<_1,_1,_1>>,
          class Permutations = Tile<Underscore,Underscore,Underscore>>
CUTE_HOST_DEVICE constexpr
auto
make_xe_tiled_mma(MMA_Atom<MMA_Op> const& mma_atom,
               MMAThrLayout     const& thr_layout   = {},
               Permutations     const& permutations = {})
{
  auto thr_layout_mnk  = append<3>(thr_layout, Layout<_1,_0>{});
  auto permutation_mnk = append<3>(permutations, _);

  return XeTiledMMA<MMA_Atom<MMA_Op>,
                  decltype(thr_layout_mnk),
                  decltype(permutation_mnk)>{mma_atom, thr_layout_mnk};
}

template <class MMA_Op,
          class MMAThrLayout = Layout<Shape<_1,_1,_1>>,
          class Permutations = Tile<Underscore,Underscore,Underscore>>
CUTE_HOST_DEVICE constexpr
auto
make_xe_tiled_mma(MMA_Op const&,
               MMAThrLayout const& thr_layout   = {},
               Permutations const& permutations = {})
{
  // Attempt to wrap in an MMA_Atom<> and forward
  return make_xe_tiled_mma(MMA_Atom<MMA_Op>{}, thr_layout, permutations);
}

template <int... I, class... Args>
CUTE_HOST_DEVICE constexpr
auto
size(XeTiledMMA<Args...> const& mma)
{
  return size<I...>(mma.get_thr_layout_vmnk());
}

}
