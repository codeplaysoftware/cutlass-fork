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
  \brief Visitor tree Softmax fusion operation for the Intel PVC epilogue
*/

#pragma once

#include "cutlass/cutlass.h"
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::fusion {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template <class STensor, uint32_t row, uint32_t SgN, class RTensor, class OutTensor>
CUTLASS_DEVICE
void group_reduce_sum_partial(STensor &stensor, RTensor &vec, OutTensor &out) {
  auto item = sycl::ext::oneapi::experimental::this_nd_item<3>();
  auto sg = item.get_sub_group();
  auto group = item.get_group();

  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < size(vec); i++) {
    vec(i) = sub_group_reduce_add(vec(i));    
  }

  auto sg_group_id = sg.get_group_id();
  auto sg_group_id_n = sg_group_id % SgN;
  auto sg_local_id = sg.get_local_id()[0];

  auto slm_base = stensor(_, _, sg_group_id / SgN);
  static constexpr auto step = SgN / IntelPVCEpilogue::SubgroupSize;
  static constexpr auto n_step = row / SgN;

  if constexpr (row < IntelPVCEpilogue::SubgroupSize) {
    if (sg_local_id < row) {
      slm_base(sg_local_id, sg_group_id_n) = vec(sg_local_id);
    }
  }
  else {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < step; i++) {
      slm_base(sg_local_id * step + i, sg_group_id_n) = vec(i + step * sg_local_id);
    }
  }

  sycl::group_barrier(group);

  Tensor s_slice = make_tensor(static_cast<decltype(slm_base) &&>(slm_base).data(), 
                        make_shape(Int<n_step>{}, Int<SgN>{}, Int<step>{}, Int<IntelPVCEpilogue::SubgroupSize>{}));
  
  if constexpr (SgN <= row) {
    auto local_vec = s_slice(_, sg_group_id_n, _, sg_local_id);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < n_step; i++) {
      auto sum = 0.f;

      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < step; j++) {
        sum += local_vec(i, j);
      }

      auto group_sum = sub_group_reduce_add(sum);

      if (sg_local_id == i) {
        s_slice(i, sg_group_id_n, 0, 0) = group_sum;
      }
    }
  }
  else {
    auto sum = 0.f;

    if (sg_group_id_n < row) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < step; j++) {
        sum += s_slice(0, sg_group_id_n, j, sg_local_id);
      }

      auto group_sum = sub_group_reduce_add(sum);

      if (sg_local_id == 0) {
        s_slice(0, sg_group_id_n,0, 0) = group_sum;
      }
    }
  }
  sycl::group_barrier(group);

  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < row; i++) {
    out(i) = slm_base(i);
  }
}

template <class STensor, uint32_t row, uint32_t SgN, class RTensor, class OutTensor>
CUTLASS_DEVICE
void group_reduce_max_partial(STensor &stensor, RTensor &vec, OutTensor &out) {
  auto item = sycl::ext::oneapi::experimental::this_nd_item<3>();
  auto sg = item.get_sub_group();
  auto group = item.get_group();

  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < size(vec); i++) {
    vec(i) = sub_group_reduce_max(vec(i));    
  }

  auto sg_group_id = sg.get_group_id();
  auto sg_group_id_n = sg_group_id % SgN;
  auto sg_local_id = sg.get_local_id()[0];

  auto slm_base = stensor(_, _, sg_group_id / SgN);
  static constexpr auto step = SgN / IntelPVCEpilogue::SubgroupSize;
  static constexpr auto n_step = row / SgN;

  if constexpr (row < IntelPVCEpilogue::SubgroupSize) {
    if (sg_local_id < row) {
      slm_base(sg_local_id, sg_group_id_n) = vec(sg_local_id);
    }
  }
  else {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < step; i++) {
      slm_base(sg_local_id * step  + i, sg_group_id_n) = vec(i + step * sg_local_id);
    }
  }

  sycl::group_barrier(group);

  Tensor s_slice = make_tensor(static_cast<decltype(slm_base) &&>(slm_base).data(), 
                        make_shape(Int<n_step>{}, Int<SgN>{}, Int<step>{}, Int<IntelPVCEpilogue::SubgroupSize>{}));

  if constexpr (SgN <= row) {
    auto local_vec = s_slice(_, sg_group_id_n, _, sg_local_id);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < n_step; i++) {
      auto local_max = local_vec(i, 0);

      CUTLASS_PRAGMA_UNROLL
      for (int j = 1; j < step; j++) {
        local_max = sycl::max(local_max, local_vec(i, j));
      }

      auto group_max = sub_group_reduce_max(local_max);

      if (sg_local_id == i) {
        s_slice(i, sg_group_id_n, 0, 0) = group_max;
      }
    }
  } 
  else {
    auto local_max = s_slice(0, sg_group_id_n, 0, sg_local_id);

    if (sg_group_id_n < row) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 1; j < step; j++) {
        local_max = sycl::max(local_max, s_slice(0, sg_group_id_n, j, sg_local_id));
      }

      auto group_max = sub_group_reduce_max(local_max);

      if (sg_local_id == 0) {
        s_slice(0, sg_group_id_n,0, 0) = group_max;
      }
    }
  }
  sycl::group_barrier(group);

  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < row; i++) {
    out(i) = slm_base(i);
  }
}

template <uint32_t SgN, class STensor, class RTensor, class OutTensor>
CUTLASS_DEVICE
auto group_reduce_sum(STensor &stensor, RTensor const &rtensor, OutTensor &out) {
  static constexpr auto row = decltype(size<0>(rtensor))::value;
  static constexpr auto col = decltype(size<1>(rtensor))::value;

  Tensor local_sum = make_tensor<float>(Shape<Int<row>>{});

  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < row; i++) {
    local_sum(i) = rtensor(i, 0);
  }

  CUTLASS_PRAGMA_UNROLL
  for (int i = 1; i < col; i++) {
    auto r_col = rtensor(_, i);

    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < row; j++) {
      local_sum(j) += r_col(j);
    }
  }
  group_reduce_sum_partial<STensor, row, SgN>(stensor, local_sum, out);
}

template <uint32_t SgN, class STensor, class RTensor, class OutTensor>
CUTLASS_DEVICE
auto group_reduce_max(STensor &stensor, RTensor const &rtensor, OutTensor &out) {
  static constexpr auto row = decltype(size<0>(rtensor))::value;
  static constexpr auto col = decltype(size<1>(rtensor))::value;

  Tensor local_max = make_tensor<float>(Shape<Int<row>>{});

  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < row; i++) {
    local_max(i) = rtensor(i, 0);
  }

  CUTLASS_PRAGMA_UNROLL
  for (int i = 1; i < col; i++) {
    auto r_col = rtensor(_, i);

    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < row; j++) {
      local_max(j) = sycl::max(local_max(j), r_col(j));
    }
  }
  group_reduce_max_partial<STensor, row, SgN>(stensor, local_max, out);
}

} // namespace detail

template <
  // int FragmentSize,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class ElementOutput,
  class ElementCompute,
  FloatRoundStyle RoundStyle
>
struct XeSoftmaxRowReduction
{
public:
  static constexpr int FragmentSize = 8;
  static constexpr auto Tile_M = get<0>(CtaTileShapeMNK{});
  static constexpr auto Tile_N = get<1>(CtaTileShapeMNK{});
  static constexpr auto Epi_M = get<0>(EpilogueTile{});
  static constexpr auto Epi_N = get<1>(EpilogueTile{});
  static constexpr auto Sg_M = Tile_M / Epi_M;
  static constexpr auto Sg_N = Tile_N / Epi_N;
  static constexpr auto Sg_Nums = Sg_M * Sg_N;
  struct SharedStorage { };

  struct Arguments { };

  struct Params { };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return {};
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    auto [M, N, K, L] = problem_shape;
    auto [tile_M, tile_N, tile_K] = CtaTileShapeMNK{};
    // Cross CTA reduction is not possible because there is no guarantee that all CTAs run
    // concurrently.
    // Cross epilogue tile reduction is possible, but re-visiting and applying reduction
    // to accumulators is only possible for the current epilogue tile.
    auto [epi_M, epi_N] = EpilogueTile{};
    return N <= tile_N;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  CUTLASS_HOST_DEVICE
  XeSoftmaxRowReduction() { }
  
  CUTLASS_HOST_DEVICE
  XeSoftmaxRowReduction(Params const& params, SharedStorage const& shared_storage)
      : params(params) { }

  Params params;

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {
    return EmptyProducerLoadCallbacks{};
  }

  template<class ArgsTuple>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(Params const& params) : params(params) {}

    // ArgsTuple args_tuple;
    Params const& params;
    template <typename ElementInput, typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE auto
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n,
          Array<ElementInput, FragmentSize> const& frg_input) {
      
      return frg_acc;
    }

    template<class STensor, class SyncFn, class VTensor>
    CUTLASS_DEVICE void
    reduce(STensor&& smem_buffer, SyncFn const& sync_fn, int epi_m, int epi_n, bool is_last_iteration, VTensor visit_results) {
      if(is_last_iteration) {
      constexpr auto vec_size = min(Epi_M, Sg_N);
      constexpr auto vec_folds = Epi_M / vec_size;

      auto smem = syclcompat::local_mem<float[Sg_Nums * vec_size]>();
      Tensor stensor = make_tensor(make_smem_ptr(smem), make_shape(Int<vec_size>{}, Int<Sg_N>{}, Int<Sg_M>{}));

      Tensor res =
          make_tensor(static_cast<decltype(visit_results) &&>(visit_results).data() - ((epi_m + 1) * (epi_n + 1) - epi_m) * FragmentSize,
                      make_shape(Int<vec_size>{}, Int<vec_folds>{}, Int<Epi_N / IntelPVCEpilogue::SubgroupSize>{}));

      CUTLASS_PRAGMA_UNROLL
      for (int loop = 0; loop < vec_folds; loop++) {
        auto loop_t = res(_, loop, _);
        Tensor group_max = make_tensor<float>(make_shape(Int<vec_size>{}));
        group_reduce_max<Sg_N>(stensor, loop_t, group_max);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < Epi_N / IntelPVCEpilogue::SubgroupSize; i++) {
          auto element_vec = loop_t(_, i);
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < vec_size; j++) {
            element_vec(j) -= group_max(j);
          }
        }
      }
      CUTLASS_PRAGMA_UNROLL
      for (int loop = 0; loop < vec_folds; loop++) {
        auto loop_t = res(_, loop, _);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < Epi_N / IntelPVCEpilogue::SubgroupSize; i++) {
          auto exp_vec = loop_t(_, i);
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < vec_size; j++) {
            exp_vec(j) = sycl::native::exp(exp_vec(j));
          }
        }
      }
      CUTLASS_PRAGMA_UNROLL
      for (int loop = 0; loop < vec_folds; loop++) {
        auto loop_t = res(_, loop, _);

        Tensor group_sum = make_tensor<float>(make_shape(Int<vec_size>{}));
        group_reduce_sum<Sg_N>(stensor, loop_t, group_sum);
      
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < Epi_N / IntelPVCEpilogue::SubgroupSize; i++) {
          auto softmax_vec = loop_t(_, i);
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < vec_size; j++) {
            softmax_vec(j) = sycl::native::divide(softmax_vec(j), group_sum(j));
          }
        }
      }
    }
    }
  };
  
  template <
  bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
  class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {
    auto args_tuple = make_tuple();
    return ConsumerStoreCallbacks<decltype(args_tuple)>(params);
  }

};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue::fusion

/////////////////////////////////////////////////////////////////////////////////////////////////
