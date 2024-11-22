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

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "online_softmax.hpp"
#include "pvc_flash_attn_mma.hpp"

namespace cutlass::gemm::kernel {

template <
  class ProblemShape,
  class CollectiveMainloop,
  class CollectiveEpilogue,
  class TileScheduler_ = void
>
class GemmUniversalAttention;

///////////////////////////////////////////////////////////////////////////////

template <
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileScheduler_
>
class GemmUniversalAttention
{
public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;

  static_assert(rank(ProblemShape{}) == 4,
    "ProblemShape{} should be <batch, num_heads, seq_len, head_size>");

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::WorkgroupTileShape;
  using WorkgroupTileShape = TileShape;
  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using ArchTag   = typename CollectiveMainloop::ArchTag;
  using ElementQ  = typename CollectiveMainloop::ElementQ;
  using StrideQ   = typename CollectiveMainloop::StrideQ;
  using ElementK  = typename CollectiveMainloop::ElementK;
  using StrideK   = typename CollectiveMainloop::StrideK;
  using ElementV  = typename CollectiveMainloop::ElementV;
  using StrideV   = typename CollectiveMainloop::StrideV;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  using SoftmaxArguments = typename flash::Softmax<ElementAccumulator>::Arguments;
  using SoftmaxParams = typename flash::Softmax<ElementAccumulator>::Params;

  static_assert(cute::is_void_v<TileScheduler_> or cute::is_same_v<TileScheduler_, PersistentScheduler>,
    "Intel PVC does not support specializing the tile scheduler.");
  using TileSchedulerTag = TileScheduler_;
  using TileScheduler = typename detail::TileSchedulerSelector<
    TileScheduler_, ArchTag, WorkgroupTileShape,
    cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::Scheduler;
  using TileSchedulerArguments = typename TileScheduler::Arguments;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementO = typename CollectiveEpilogue::ElementO;
  using StrideO  = typename CollectiveEpilogue::StrideO;
  using ElementLSE = typename CollectiveEpilogue::ElementLSE;
  using StrideLSE  = typename CollectiveEpilogue::StrideLSE;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;
  static_assert(cute::is_same_v<ElementAccumulator, typename CollectiveEpilogue::ElementAccumulator>,
    "Mainloop and epilogue do not agree on accumulator value type.");

  // MSVC requires the cast to fix a warning-as-error.
  static constexpr int SharedStorageSize = 0;

  static constexpr bool CausalMask = CollectiveMainloop::CausalMask;
  static constexpr int SubgroupSize = CollectiveMainloop::SubgroupSize; // sub_group size
  static constexpr uint32_t MaxThreadsPerBlock = CollectiveMainloop::MaxThreadsPerBlock;
  using MmaAtomShape = typename CollectiveMainloop::MmaAtomShape;
  using SubgroupTileShape = typename CollectiveMainloop::SubgroupTileShape;

  static constexpr int BLK_M = CollectiveMainloop::BLK_M;
  static constexpr int BLK_N = CollectiveMainloop::BLK_N;
  static constexpr int BLK_K = CollectiveMainloop::BLK_K;

  static constexpr int ATOM_M = CollectiveMainloop::ATOM_M;
  static constexpr int ATOM_N = CollectiveMainloop::ATOM_N;
  static constexpr int ATOM_K = CollectiveMainloop::ATOM_K;

  static constexpr int SG_M = CollectiveMainloop::SG_M;
  static constexpr int SG_N = CollectiveMainloop::SG_N;
  static constexpr int SG_K = CollectiveMainloop::SG_K;

  static constexpr int Vec = (get<0>(MmaAtomShape()) * get<1>(MmaAtomShape())) / SubgroupSize;
  static constexpr int FragsM = get<0>(SubgroupTileShape{}) / get<0>(MmaAtomShape());
  static constexpr int FragsN = get<1>(SubgroupTileShape{}) / get<1>(MmaAtomShape());

  // Kernel level shared memory storage
  struct SharedStorage {
    using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;
    EpilogueTensorStorage epilogue;
  };

  // Device side arguments
  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    SoftmaxArguments softmax{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
  };

  // Kernel entry point API
  struct Params {
    GemmUniversalMode mode;
    ProblemShape problem_shape;
    MainloopParams mainloop;
    SoftmaxArguments softmax;
    EpilogueParams epilogue;
  };

  //
  // Methods
  //

  // Convert to underlying arguments. In this case, a simple copy for the aliased type.
  static
  Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    (void) workspace;
    return {
      args.mode,
      args.problem_shape,
      CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, workspace),
      flash::Softmax<ElementAccumulator>::to_underlying_arguments(args.softmax),
      CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, workspace)
    };
  }

  static bool
  can_implement(Arguments const& args) {
    bool mode_implementable = args.mode == GemmUniversalMode::kGemm or
          (args.mode == GemmUniversalMode::kBatched && rank(ProblemShape{}) == 4);
    return mode_implementable && TileScheduler::can_implement(args.scheduler);
  }

  static int
  get_workspace_size(Arguments const& args) {
    return 0;
  }

  static
  cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr, 
    CudaHostAdapter* cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  static dim3
  get_grid_shape(Params const& params) {
    return dim3(
            cute::size(cute::ceil_div(cute::shape<3>(params.problem_shape), cute::shape<1>(WorkgroupTileShape{}))),
            cute::size(cute::ceil_div(cute::shape<2>(params.problem_shape), cute::shape<0>(WorkgroupTileShape{}))),
            cute::size(cute::shape<0>(params.problem_shape) * cute::shape<1>(params.problem_shape))
    );
  }

  static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  CUTLASS_DEVICE
  void
  operator()(Params const& params, char* smem_buf) {
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    // Preconditions
    CUTE_STATIC_ASSERT(is_static<WorkgroupTileShape>::value);

    // Separate out problem shape for convenience
    auto batch = get<0>(params.problem_shape);
    auto num_heads = get<1>(params.problem_shape);
    auto seq_len = get<2>(params.problem_shape);
    auto head_size = get<3>(params.problem_shape);

    // Preconditions
    static_assert(cute::rank(StrideQ{}) == 3, "StrideQ must be rank-4: [batch, num_heads, seq_len, head_size].");
    static_assert(cute::rank(StrideK{}) == 3, "StrideK must be rank-4: [batch, num_heads, seq_len, head_size].");
    static_assert(cute::rank(StrideV{}) == 3, "StrideV must be rank-4: [batch, num_heads, seq_len, head_size].");

    int thread_idx = int(ThreadIdxX());
    int sub_group_id = thread_idx / SubgroupSize;
    constexpr auto workgroup_shape = WorkgroupTileShape{};                                                  // (SUB_M,SUB_N,SUB_K)
    constexpr auto subgroup_shape = SubgroupTileShape{};                                                  // (SUB_M,SUB_N,SUB_K)

    auto blk_shape = TileShape{};
    auto blk_m_coord = BlockIdxY();
    auto blk_n_coord = BlockIdxX();
    auto blk_l_coord = BlockIdxZ();
    auto blk_coord_mnkl = make_coord(blk_m_coord, blk_n_coord, _, blk_l_coord);

    Tensor mQ_mkl = make_tensor(make_gmem_ptr(static_cast<ElementQ const*>(nullptr)), make_shape(seq_len, head_size, batch * num_heads), StrideQ{});   //(m,k,l)
    Tensor mK_nkl = make_tensor(make_gmem_ptr(static_cast<ElementK const*>(nullptr)), make_shape(seq_len, head_size, batch * num_heads), StrideK{});   //(n,k,l)
    Tensor mV_nkl = make_tensor(make_gmem_ptr(static_cast<ElementV const*>(nullptr)), make_shape(head_size, seq_len, batch * num_heads), StrideV{});   //(n,k,l)

    Tensor mQ_mk = mQ_mkl(_,_,blk_l_coord);                                                                        // (m,k)
    Tensor mK_nk = mK_nkl(_,_,blk_l_coord);                                                                        // (n,k)
    Tensor mV_nk = mV_nkl(_,_,blk_l_coord);                                                                        // (n,k)

    auto gQ = local_tile(mQ_mk, blk_shape, make_coord(blk_m_coord, 0, _), Step<_1,  X, _1>{});

    const int seq_coord = BlockIdxY() * BLK_M + (sub_group_id / ATOM_N) * SG_M;
    const int head_size_coord = BlockIdxX() * BLK_N + (sub_group_id % ATOM_N) * SG_N;
    const int l_coord = BlockIdxZ();

    // Compute tile residues for predication
    auto m_max_coord = seq_len - get<0>(subgroup_shape) * seq_coord;                             // M - SUB_M * m_coord
    auto n_max_coord = seq_len - get<1>(subgroup_shape) * seq_coord;                             // N - SUB_N * n_coord
    auto k_residue   = head_size - get<2>(subgroup_shape) * (head_size / get<2>(subgroup_shape));        // K - SUB_K * k_coord_max
    auto residue_mnk = make_tuple(m_max_coord, n_max_coord, k_residue);

    // Allocate the tiled_mma and the accumulators for the (M,N) subgroup_shape
    TiledMma tiled_mma;

    Tensor out_reg = partition_fragment_C(tiled_mma, take<0,2>(blk_shape)); 

    Tensor max_reg = make_tensor<ElementAccumulator>(Shape<Int<Vec>, Int<FragsM>>{});
    Tensor sum_reg = make_tensor<ElementAccumulator>(Shape<Int<Vec>, Int<FragsM>>{});

    fill(max_reg, -INFINITY);
    clear(sum_reg);
    clear(out_reg);

    // Perform the collective scoped MMA
    CollectiveMainloop collective_mma;

    const int causal_seq_len = seq_coord + get<0>(subgroup_shape);
    const int non_causal_seq_len = seq_len;

    const int nblock_limit = CausalMask ? cute::ceil_div(causal_seq_len, get<1>(subgroup_shape)) 
                                : cute::ceil_div(non_causal_seq_len, get<1>(subgroup_shape));

    const int item_id = thread_idx % SubgroupSize;

    // loop over K and V, perform fused attention + online softmax
    for (int nblock = 0, load_idx = 0; nblock < nblock_limit; nblock++,
        load_idx += get<1>(subgroup_shape)) {
      // 1) Load K (performed inside mmaQK)
      // 2) Create Tensor S
      auto gK = local_tile(mK_nk, blk_shape, make_coord(0, 0, _), Step< X, _1, _1>{});
      Tensor tSr = make_tensor<ElementAccumulator>(Shape<Int<Vec>, Int<FragsM>, Int<FragsN>>{});
      clear(tSr);
      // 3) Perform GEMM S = Q*K
      auto tile_coord_QK = make_coord(seq_coord, load_idx, _, blk_l_coord);
      collective_mma.mmaQK(tile_coord_QK, tSr, gQ, gK, tSr, head_size / get<1>(subgroup_shape), params.mainloop);

      // Apply causal mask
      if constexpr (CausalMask) {
        // mask the elements of each tile where j > i
        int col_idx = item_id + load_idx;
        CUTLASS_PRAGMA_UNROLL
        for(int n = 0; n < FragsN; n++, col_idx += get<1>(MmaAtomShape())) {
          CUTLASS_PRAGMA_UNROLL
          for(int m = 0; m < FragsM; m++) {
            int row_idx = m * Vec + seq_coord;
            CUTLASS_PRAGMA_UNROLL
            for(int row = 0; row < Vec; row++, row_idx++) {
              if(col_idx > row_idx)
                tSr(row, m, n) = -INFINITY;
            }
          }
        }
      }

      if (nblock == 0)
        flash::Softmax<ElementAccumulator>::template run<true, CausalMask, Vec, FragsM, FragsN>(tSr, 
                                                          max_reg, sum_reg, out_reg, params.softmax);
      else
        flash::Softmax<ElementAccumulator>::template run<false, CausalMask, Vec, FragsM, FragsN>(tSr, 
                                                          max_reg, sum_reg, out_reg, params.softmax);
      // 7) Convert S to P (FP32 -> BF16)
      Tensor tPr = make_tensor<typename TiledMma::ValTypeA>(shape(tSr));
      CUTLASS_PRAGMA_UNROLL
      for (int p_idx = 0; p_idx < size(tPr); p_idx++) {
        tPr(p_idx) = static_cast<typename TiledMma::ValTypeA>(tSr(p_idx));
      }

      // 8) Scale out_reg with l
      // 10) Perform GEMM O = 
      auto gV = local_tile(mV_nk, blk_shape, make_coord(0, 0, _), Step< X, _1, _1>{});
      auto tile_coord_PV = make_coord(0, head_size_coord, _, blk_l_coord);
      collective_mma.mmaPV(tile_coord_PV, out_reg, tPr, gV, out_reg, 1, nblock, params.mainloop);
    }

    // Reduce the sum of exponents across the subgroup before scaling/normalizing output
    flash::SumOp<ElementAccumulator> op;
    flash::Softmax<ElementAccumulator>::template subgroup_allreduce<false, Vec, FragsM, FragsN>(sum_reg, op);

    CollectiveEpilogue epilogue{params.epilogue, shared_storage.epilogue};

    epilogue(
      params.problem_shape, 
      blk_coord_mnkl, 
      out_reg, 
      max_reg, 
      sum_reg, 
      tiled_mma, 
      params.softmax.scale);
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel
