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
/*! \file
  \brief Functor performing online softmax.
*/

#pragma once

#include <sycl/sycl.hpp>
#include "cutlass/cutlass.h"

#ifdef __SYCL_DEVICE_ONLY__
#define SYCL_DEVICE_OCL_FMA(x, y) SYCL_EXTERNAL x
#else
#define SYCL_DEVICE_OCL_FMA(x, y) \
  inline x                        \
  {                               \
    assert(false);                \
    return (y)0;                  \
  }
#endif

#define EXP sycl::native::exp
#define DIV sycl::native::divide

SYCL_DEVICE_OCL_FMA(float sub_group_reduce_add(float i), float);
SYCL_DEVICE_OCL_FMA(float sub_group_reduce_max(float i), float);

#undef SYCL_DEVICE_OCL_FMA

namespace flash
{

  template <typename T>
  struct MaxOp
  {
    CUTLASS_DEVICE T
    operator()(T const &x, T const &y) { return sycl::max(x, y); }
  };

  template <typename T>
  struct SumOp
  {
    CUTLASS_DEVICE T
    operator()(T const &x, T const &y) { return x + y; }
  };

  struct FirstBlockSoftmax
  {
    template <
        bool CheckInf,
        bool is_first,
        int SizeA,
        int SizeB,
        int SizeC, typename Element, typename FragSum, typename FragMax, typename FragMax_Prev, typename FragOut>
    CUTLASS_DEVICE static typename std::enable_if<is_first>::type run(const Element, FragSum &, FragMax &, FragMax_Prev &, FragOut &) {}

    template <
        bool CheckInf,
        bool is_first,
        int SizeA,
        int SizeB,
        int SizeC, typename Element, typename FragSum, typename FragMax, typename FragMax_Prev, typename FragOut>
    CUTLASS_DEVICE static typename std::enable_if<!is_first>::type run(const Element scale, FragSum &sum, FragMax &max, FragMax_Prev &max_prev, FragOut &out)
    {
      CUTLASS_PRAGMA_UNROLL
      for (int x = 0; x < SizeA; x++) // V
      {
        CUTLASS_PRAGMA_UNROLL
        for (int y = 0; y < SizeB; y += 4) // M
        {
          Element curr_max_scaled = max(x, y) * scale;
          Element curr_max_scale1 = max(x, y + 1) * scale;
          Element curr_max_scaled2 = max(x, y + 2) * scale;
          Element curr_max_scale3 = max(x, y + 3) * scale;
          if (CheckInf && max(x, y) == -INFINITY)
          {
            curr_max_scaled = 0.f;
          }
          if (CheckInf && max(x, y + 1) == -INFINITY)
          {
            curr_max_scale1 = 0.f;
          }
          if (CheckInf && max(x, y + 2) == -INFINITY)
          {
            curr_max_scaled2 = 0.f;
          }
          if (CheckInf && max(x, y + 3) == -INFINITY)
          {
            curr_max_scale3 = 0.f;
          }

          const Element eq = (max_prev(x, y) * scale - curr_max_scaled);
          const Element eq1 = (max_prev(x, y + 1) * scale - curr_max_scale1);
          const Element eq2 = (max_prev(x, y + 2) * scale - curr_max_scaled2);
          const Element eq3 = (max_prev(x, y + 3) * scale - curr_max_scale3);
          const Element curr_scale = sycl::native::exp2(eq);
          const Element curr_scale1 = sycl::native::exp2(eq1);
          const Element curr_scale2 = sycl::native::exp2(eq2);
          const Element curr_scale3 = sycl::native::exp2(eq3);
          sum(x, y) *= curr_scale;
          sum(x, y + 1) *= curr_scale1;
          sum(x, y + 2) *= curr_scale2;
          sum(x, y + 3) *= curr_scale3;
          CUTLASS_PRAGMA_UNROLL
          for (int z = 0; z < SizeC; z++) // N
          {
            out(x, y, z) *= curr_scale;
            out(x, y + 1, z) *= curr_scale1;
            out(x, y + 2, z) *= curr_scale2;
            out(x, y + 3, z) *= curr_scale3;
          }
        }
      }
    }
  };

  template <typename Element>
  struct Softmax
  {
    struct Arguments
    {
      Element scale;
    };

    using Params = Arguments;

    static constexpr Params
    to_underlying_arguments(Arguments const &args)
    {
      Arguments x{static_cast<Element>(args.scale) * static_cast<Element>(M_LOG2E)};
      return x;
    }

    template <
        bool CheckInf,
        int SizeA,
        int SizeB,
        int SizeC,
        class FragAcc,
        class FragMax>
    CUTLASS_DEVICE static constexpr void scale_exp_log2(FragAcc &acc, FragMax const max, Element const scale)
    {
      CUTLASS_PRAGMA_UNROLL
      for (int x = 0; x < SizeA; x++)
      { // size A =8
        CUTLASS_PRAGMA_UNROLL
        for (int y = 0; y < SizeB; y++)
        { // size B = 4
          Element max_scaled = max(x, y) * scale;
          if (CheckInf && max(x, y) == -INFINITY)
          {
            max_scaled = 0.f;
          }

          CUTLASS_PRAGMA_UNROLL
          for (int z = 0; z < SizeC; z += 2)
          { // size C = 2
            Element eq0 = (acc(x, y, z) * scale - max_scaled);
            Element eq1 = (acc(x, y, z + 1) * scale - max_scaled);
            acc(x, y, z) = sycl::native::exp2(eq0);
            acc(x, y, z + 1) = sycl::native::exp2(eq1);
          }
        }
      }
    }

    template <
        int SizeA,
        int SizeB,
        int SizeC,
        class FragSrc,
        class FragDst,
        class Op>
    CUTLASS_DEVICE static void workitem_reduce(FragSrc const &src, FragDst &dst, Op op)
    {
      // reduction per work item
      CUTLASS_PRAGMA_UNROLL
      for (int x = 0; x < SizeA; x++)
      {
        CUTLASS_PRAGMA_UNROLL
        for (int y = 0; y < SizeB; y++)
        {
          dst(x, y) = op(dst(x, y), src(x, y, 0));
          CUTLASS_PRAGMA_UNROLL
          for (int z = 1; z < SizeC; z++)
          {
            dst(x, y) = op(dst(x, y), src(x, y, z));
          }
        }
      }
    }

    template <
        int SizeA,
        int SizeB,
        int SizeC,
        class FragDst,
        class Op>
    CUTLASS_DEVICE static void subgroup_allreduce(FragDst &dst, Op op)
    {
      // reduce across the sub_group to get the final output
      auto sg = syclcompat::get_nd_item<1>().get_sub_group();
      CUTLASS_PRAGMA_UNROLL
      for (int x = 0; x < SizeA; x++)
      {
        CUTLASS_PRAGMA_UNROLL
        for (int y = 0; y < SizeB; y++)
        {
          dst(x, y) = std::is_same_v<Op, SumOp<float>> ? sub_group_reduce_add(dst(x, y)) : sub_group_reduce_max(dst(x, y));
        }
      }
    }

    template <
        int SizeA,
        int SizeB,
        int SizeC,
        class FragSrc,
        class FragDst,
        class Op>
    CUTLASS_DEVICE static void reduce(FragSrc const &src, FragDst &dst, Op op)
    {
      // reduce across all the N tiles in shape <VecC, FragsM, FragsN>
      workitem_reduce<SizeA, SizeB, SizeC>(src, dst, op);
      subgroup_allreduce<SizeA, SizeB, SizeC>(dst, op);
    }

    template <
        int SizeA,
        int SizeB,
        int SizeC,
        class FragSrc,
        class FragMax>
    CUTLASS_DEVICE static void reduce_max(FragSrc const &src, FragMax &max)
    {
      MaxOp<Element> max_op;
      reduce<SizeA, SizeB, SizeC>(src, max, max_op);
    }

    template <
        int SizeA,
        int SizeB,
        int SizeC,
        class FragSrc,
        class FragSum>
    CUTLASS_DEVICE static void reduce_sum(FragSrc const &src, FragSum &sum)
    {
      SumOp<Element> sum_op;
      workitem_reduce<SizeA, SizeB, SizeC>(src, sum, sum_op);
    }

    template <
        bool CheckInf,
        int SizeA,
        int SizeB,
        int SizeC,
        class FragAcc,
        class FragMax,
        class FragSum,
        class FragOut>
    CUTLASS_DEVICE static void run(bool is_first, FragAcc &frag_s, FragMax &max, FragSum &sum, FragOut &out, Params const &params)
    {
      cute::Tensor max_prev = cute::make_fragment_like(max);
      cute::copy(max, max_prev);
      reduce_max<SizeA, SizeB, SizeC>(frag_s, max);
      (is_first) ? FirstBlockSoftmax::template run<CheckInf, true, SizeA, SizeB, SizeC>(params.scale, sum, max, max_prev, out)
                 : FirstBlockSoftmax::template run<CheckInf, false, SizeA, SizeB, SizeC>(params.scale, sum, max, max_prev, out);
      scale_exp_log2<CheckInf, SizeA, SizeB, SizeC>(frag_s, max, params.scale);
      reduce_sum<SizeA, SizeB, SizeC>(frag_s, sum);
    }

    Params params;
  };

}
