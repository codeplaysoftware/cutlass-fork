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

namespace flash {

template<typename T>
struct MaxOp {
    CUTLASS_DEVICE T
    operator()(T const & x, T const & y) { return x > y ? x : y; }
};

template<typename T>
struct SumOp {
    CUTLASS_DEVICE T
    operator()(T const & x, T const & y) { return x + y; }
};

template <typename Element>
struct Softmax {
    struct Arguments {
        Element scale;
    };

    using Params = Arguments;

    static constexpr Params
    to_underlying_arguments(Arguments const& args) {
        Arguments x{static_cast<Element>(args.scale) * static_cast<Element>(M_LOG2E)};
        return x;
    }

    template <
    bool CheckInf,
    int SizeA,
    int SizeB,
    int SizeC,
    class FragAcc,
    class FragMax
    >
    CUTLASS_DEVICE static constexpr 
    void scale_exp_log2(FragAcc &acc, FragMax const max, Element const scale) {
        CUTLASS_PRAGMA_UNROLL
        for(int x = 0; x < SizeA; x++) {
            CUTLASS_PRAGMA_UNROLL
            for(int y = 0; y < SizeB; y++) {
                Element max_scale = !CheckInf ? max(x, y) : max(x, y) == -INFINITY ? Element{0} : max(x, y);
                CUTLASS_PRAGMA_UNROLL
                for(int z = 0; z < SizeC; z++) {
                    acc(x, y, z) = sycl::native::exp2((acc(x, y, z) - max_scale) * scale);
                }
            }
        }
    }

    template <
    bool zero_init,
    int SizeA,
    int SizeB,
    int SizeC,
    class FragSrc,
    class FragDst,
    class Op
    >
    CUTLASS_DEVICE static void workitem_reduce(FragSrc const &src, FragDst &dst, Op op) {
        // reduction per work item
        CUTLASS_PRAGMA_UNROLL
        for(int x = 0; x < SizeA; x++) {
            CUTLASS_PRAGMA_UNROLL
            for(int y = 0; y < SizeB; y++) {
                dst(x, y) = zero_init ? src(x, y, 0) : op(dst(x, y), src(x, y, 0));
                CUTLASS_PRAGMA_UNROLL
                for(int z = 1; z < SizeC; z++) {
                    dst(x, y) = op(dst(x, y), src(x, y, z));
                }
            }
        }
    }

    template <
    bool zero_init,
    int SizeA,
    int SizeB,
    int SizeC,
    class FragDst,
    class Op
    >
    CUTLASS_DEVICE static void subgroup_allreduce(FragDst &dst, Op op) {
        // reduce across the sub_group to get the final output
        auto sg = syclcompat::get_nd_item<1>().get_sub_group();
        CUTLASS_PRAGMA_UNROLL
        for(int x = 0; x < SizeA; x++) {
            CUTLASS_PRAGMA_UNROLL
            for(int y = 0; y < SizeB; y++) {
                CUTLASS_PRAGMA_UNROLL
                for(uint laneMask = 8; laneMask >= 1; laneMask /= 2) {
                    dst(x,y) = op(dst(x, y), syclcompat::permute_sub_group_by_xor(sg, dst(x, y), laneMask, 16));
                }
            }
        }
    }

    template <
    bool zero_init,
    int SizeA,
    int SizeB,
    int SizeC,
    class FragSrc,
    class FragDst,
    class Op
    >
    CUTLASS_DEVICE static void reduce(FragSrc const &src, FragDst &dst, Op op) {
        // reduce across all the N tiles in shape <VecC, FragsM, FragsN>
        workitem_reduce<zero_init, SizeA, SizeB, SizeC>(src, dst, op);
        subgroup_allreduce<zero_init, SizeA, SizeB, SizeC>(dst, op);
    }

    template <
    bool zero_init,
    int SizeA,
    int SizeB,
    int SizeC,
    class FragSrc,
    class FragMax
    >
    CUTLASS_DEVICE static void reduce_max(FragSrc const &src, FragMax& max) {
        MaxOp<Element> max_op;
        reduce<zero_init, SizeA, SizeB, SizeC>(src, max, max_op);
    }

    template <
    bool zero_init,
    int SizeA,
    int SizeB,
    int SizeC,
    class FragSrc,
    class FragSum
    >
    CUTLASS_DEVICE static void reduce_sum(FragSrc const &src, FragSum& sum) {
        SumOp<Element> sum_op;
        workitem_reduce<zero_init, SizeA, SizeB, SizeC>(src, sum, sum_op);
    }

    template <
    bool is_first,
    bool CheckInf,
    int SizeA,
    int SizeB,
    int SizeC,
    class FragAcc,
    class FragMax,
    class FragSum,
    class FragOut
    > 
    CUTLASS_DEVICE static typename std::enable_if<is_first>::type
    run(FragAcc &frag_s, FragMax& max, FragSum& sum, FragOut&, Params const &params) {
        reduce_max<is_first, SizeA, SizeB, SizeC>(frag_s, max);
        scale_exp_log2<CheckInf, SizeA, SizeB, SizeC>(frag_s, max, params.scale);
        reduce_sum<is_first, SizeA, SizeB, SizeC>(frag_s, sum);
    }

    template <
    bool is_first,
    bool CheckInf,
    int SizeA,
    int SizeB,
    int SizeC,
    class FragAcc,
    class FragMax,
    class FragSum,
    class FragOut
    > 
    CUTLASS_DEVICE static typename std::enable_if<!is_first>::type
    run(FragAcc &frag_s, FragMax& max, FragSum& sum, FragOut &out, Params const &params) {
        cute::Tensor max_prev = cute::make_fragment_like(max);
        cute::copy(max, max_prev);
        reduce_max<is_first, SizeA, SizeB, SizeC>(frag_s, max);

        CUTLASS_PRAGMA_UNROLL
        for(int x = 0; x < SizeA; x++) {
            CUTLASS_PRAGMA_UNROLL
            for(int y = 0; y < SizeB; y++) {
                Element curr_max = !CheckInf ? max(x, y) : max(x, y) == -INFINITY ? 0.0f : max(x, y);
                Element curr_scale = sycl::native::exp2((max_prev(x, y) - curr_max) * params.scale);
                sum(x, y) *= curr_scale;
                CUTLASS_PRAGMA_UNROLL
                for(int z = 0; z < SizeC; z++) {
                    out(x, y, z) *= curr_scale;
                }
            }
        }

        scale_exp_log2<CheckInf, SizeA, SizeB, SizeC>(frag_s, max, params.scale);
        reduce_sum<is_first, SizeA, SizeB, SizeC>(frag_s, sum);
    }

    Params params;
};
}