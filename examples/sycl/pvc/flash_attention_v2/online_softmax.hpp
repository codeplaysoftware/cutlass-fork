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

// TODO:: Temporary using OpenCL function as sycl_reduce_over_group spills on sycl::maximum<>() operation
SYCL_DEVICE_OCL_FMA(float sub_group_reduce_add(float i), float);
SYCL_DEVICE_OCL_FMA(float sub_group_reduce_max(float i), float);

#undef SYCL_DEVICE_OCL_FMA

namespace flash
{

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
        bool CausalMask,
        int Vec,
        int FragsM,
        int FragsN,
        class FragAcc,
        class FragMax,
        class FragSum
        >
    CUTLASS_DEVICE static constexpr void scale_exp_log2(FragAcc &frag_s, FragMax const max,  FragSum& sum, Params const params)
    {
      auto g = syclcompat::get_nd_item<1>().get_sub_group();
      // FIXME: The performance would vary depending on the location we scale the max 
      // for causal true or false while code generated is the same. Temporarily diverging 
      // the path to avoid performance divergence
      const auto max_scale = (CausalMask)? max * params.scale : max;
      
      CUTLASS_PRAGMA_UNROLL
      for (int y = 0; y < FragsM; y++)
      { 
        CUTLASS_PRAGMA_UNROLL
        for (int x = 0; x < Vec; x+=2)
        { 
          int indx = ((y * Vec) + x);
          // FIXME: The performance would vary depending on the location we scale the max 
          // for causal true or false while code generated is the same. Temporarily diverging 
          // the path to avoid performance divergence
          const auto max_scale_bcast = (CausalMask) ?(group_broadcast(g, max_scale, (indx / 2) % 16)) 
                                                    : (group_broadcast(g, max_scale, (indx / 2) % 16))* params.scale;
          CUTLASS_PRAGMA_UNROLL 
          for (int z = 0; z < FragsN; z++)
          {         
            auto base_indx = indx + (z * Vec * FragsM);
            sycl::vec<Element, 2> eq = {frag_s(base_indx) - max_scale_bcast[0],
                                        frag_s(base_indx + 1) - max_scale_bcast[1]};
            frag_s(base_indx) = sycl::native::exp2(eq[0]);
            frag_s(base_indx + 1) = sycl::native::exp2(eq[1]);
            sum(indx) += frag_s(base_indx); 
            sum(indx + 1) += frag_s(base_indx + 1);
          }
        }
      }
    }

    template <
        int Vec,
        int FragsM,
        int FragsN,
        class FragSrc,
        class FragMax>
    CUTLASS_DEVICE static void reduce_max(FragSrc &src, FragMax& max, Params const & params)
    {       
      auto g = syclcompat::get_nd_item<1>().get_sub_group();
      sycl::vec<Element,2> maxptr[Vec * FragsM / 2];
      CUTLASS_PRAGMA_UNROLL
      for (int y = 0; y < FragsM; y++)
      {
        CUTLASS_PRAGMA_UNROLL
        for (int x = 0; x < Vec; x += 2)
        {
          int indx = ((y * Vec) + x) / 2;
          maxptr[indx] = group_broadcast(g, max, indx % 16);
          CUTLASS_PRAGMA_UNROLL
          for (int z = 0; z < FragsN; z++)
          { 
            auto base_indx = 2 * indx + (z * Vec * FragsM); 
            maxptr[indx] = sycl::max(maxptr[indx], sycl::vec<Element,2>{src(base_indx), src(base_indx + 1)});
            src(base_indx) *= params.scale;
            src(base_indx + 1) *= params.scale;
          }
          maxptr[indx] = {sub_group_reduce_max(maxptr[indx][0]), sub_group_reduce_max(maxptr[indx][1])};
        }
      }
      int idx = g.get_local_id()[0];
      max = maxptr[idx];
    }
    template <
        bool CausalMask,
        int Vec,
        int FragsM,
        int FragsN,
        class FragAcc,
        class FragMax,
        class FragSum,
        class FragOut>
    CUTLASS_DEVICE static void run(bool is_first, FragAcc &frag_s, FragMax &max, FragSum &sum, FragOut &out, Params const  &params)
    {
      auto max_prev = max;
      reduce_max<Vec, FragsM, FragsN>(frag_s, max, params);  
      static_assert(Vec * FragsM == 32 && Vec >= 2, "For online softmax Vec size must be 8. otherwise, the number of reg_max per workitem should be adoted accordingly.");
      if(!is_first) {
        auto g = syclcompat::get_nd_item<1>().get_sub_group();
        sycl::vec<Element, 2> max_scale{max * params.scale};
        sycl::vec<Element, 2> exp_scale{sycl::native::exp2(max_prev * params.scale - max_scale)};
        int idx = g.get_local_id()[0];      
        CUTLASS_PRAGMA_UNROLL
        for (int y = 0; y < FragsM; y++ )
        { 
          CUTLASS_PRAGMA_UNROLL
          for (int x =0; x < Vec; x += 2)
          {
            int indx = ((y * Vec) + x);
            auto max_scale_bcast = group_broadcast(g, max_scale, (indx/2) % 16);
            auto exp_scale_bcast = group_broadcast(g, exp_scale, (indx/2) % 16);
            sum(indx) *= exp_scale_bcast[0];
            sum(indx + 1) *= exp_scale_bcast[1];
            CUTLASS_PRAGMA_UNROLL
            for (int z = 0; z < FragsN; z++)
            {
              auto base_indx = indx + (z * Vec * FragsM); 
              out(base_indx) *= exp_scale_bcast[0];
              out(base_indx + 1 ) *= exp_scale_bcast[1];
              frag_s(base_indx) = sycl::native::exp2((frag_s(base_indx) - max_scale_bcast[0]));
              frag_s(base_indx + 1) = sycl::native::exp2((frag_s(base_indx + 1) - max_scale_bcast[1]));
              sum(indx) += frag_s(base_indx); 
              sum(indx + 1) += frag_s(base_indx + 1);
            }
          }
        }
      } else {
        scale_exp_log2<CausalMask, Vec, FragsM, FragsN>(frag_s, max, sum, params);
      }
    }
    Params params;
  };
}
