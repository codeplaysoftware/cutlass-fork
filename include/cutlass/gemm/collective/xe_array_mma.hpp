/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
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
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/tensor_predicate.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;
/////////////////////////////////////////////////////////////////////////////////////////////////

template <int Stages, class Schedule, class TileShape_, class ElementA_, class StrideA_, class ElementB_, class StrideB_,
          class TiledMma_, class GmemTiledCopyA_, class SmemLayoutAtomA_, class SmemCopyAtomA_, class TransformA_,
          class GmemTiledCopyB_, class SmemLayoutAtomB_, class SmemCopyAtomB_, class TransformB_>
struct CollectiveMma<MainloopIntelPVCGroup<Stages, Schedule>, TileShape_, ElementA_, StrideA_, ElementB_, StrideB_, TiledMma_,
                     GmemTiledCopyA_, SmemLayoutAtomA_, SmemCopyAtomA_, TransformA_, GmemTiledCopyB_, SmemLayoutAtomB_,
                     SmemCopyAtomB_, TransformB_> {
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopIntelPVCGroup<Stages, Schedule>;
  using WorkgroupTileShape = TileShape_;
  using ElementA = ElementA_;
  using StrideA = StrideA_;
  using InternalStrideA = cute::remove_pointer_t<StrideA>;
  using ElementB = ElementB_;
  using StrideB = StrideB_;
  using InternalStrideB = cute::remove_pointer_t<StrideB>;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;

  static_assert(platform::is_same<ElementA, ElementB>::value, "MainloopIntelPVCArray requires that A and B have same type.");

  static constexpr int SubgroupSize = DispatchPolicy::SubgroupSize;

  using MmaAtomShape = typename TiledMma::AtomShape_MNK;

  static constexpr auto BLK_M = get<0>(WorkgroupTileShape{});
  static constexpr auto BLK_N = get<1>(WorkgroupTileShape{});
  static constexpr auto BLK_K = get<2>(WorkgroupTileShape{});

  static constexpr auto ATOM_M = get<1>(typename TiledMma::ThrLayoutVMNK{}.shape());
  static constexpr auto ATOM_N = get<2>(typename TiledMma::ThrLayoutVMNK{}.shape());
  static constexpr auto ATOM_K = get<3>(typename TiledMma::ThrLayoutVMNK{}.shape());

  static constexpr auto SG_M = ceil_div(BLK_M, ATOM_M);
  static constexpr auto SG_N = ceil_div(BLK_N, ATOM_N);
  static constexpr auto SG_K = ceil_div(BLK_K, ATOM_K);
  using SubgroupTileShape = Shape<decltype(SG_M), decltype(SG_N), decltype(SG_K)>;

  static constexpr bool is_A_transposed = cute::detail::is_transpose_load<GmemTiledCopyA_>;
  static constexpr bool is_B_transposed = cute::detail::is_transpose_load<GmemTiledCopyB_>;
  static constexpr size_t cacheline_bytes = 64;
  // min(32,32)-> 32 (256, 32) -> 32
  static constexpr auto block_size_w_a = cute::min(is_A_transposed ? BLK_M : BLK_K, cacheline_bytes / sizeof(ElementA));
  // 1 -> trans 256/32 = 8
  static constexpr auto nums_block_w_a = ceil_div(is_A_transposed ? BLK_M : BLK_K, block_size_w_a);
  // min(32, 32) ->32  (256, 32)
  static constexpr auto block_size_w_b = cute::min(is_B_transposed ? BLK_K : BLK_N, cacheline_bytes / sizeof(ElementB));
  // 8  -> trans 1
  static constexpr auto nums_block_w_b = ceil_div(is_B_transposed ? BLK_K : BLK_N, block_size_w_b);
  // 32
  static constexpr auto Total_SG = ATOM_N * ATOM_M * ATOM_K;

  // shape<32,1> / trans shap<4,8>
  using PrefetchAThrShape =
      Shape<Int<Total_SG / cute::gcd(Total_SG, nums_block_w_a)>, Int<cute::gcd(Total_SG, nums_block_w_a)>>;
  // shape<4,8> / trans shap<32,1>
  using PrefetchBThrShape =
      Shape<Int<Total_SG / cute::gcd(Total_SG, nums_block_w_b)>, Int<cute::gcd(Total_SG, nums_block_w_b)>>;
  // 8x32
  using PrefetchATileSize = decltype(ceil_div(
      Shape<Int<is_A_transposed ? BLK_K : BLK_M>, Int<is_A_transposed ? BLK_M : BLK_K>>{}, PrefetchAThrShape{}));
  // 8x32
  using PrefetchBTileSize = decltype(ceil_div(
      Shape<Int<is_B_transposed ? BLK_N : BLK_K>, Int<is_B_transposed ? BLK_K : BLK_N>>{}, PrefetchBThrShape{}));

  static constexpr uint32_t MaxThreadsPerBlock = size(TiledMma{});

  using traits_load_A = Copy_Traits<GmemTiledCopyA, InternalStrideA>;
  using atom_load_A = Copy_Atom<traits_load_A, ElementA>;

  using traits_load_B = Copy_Traits<GmemTiledCopyB, InternalStrideB>;
  using atom_load_B = Copy_Atom<traits_load_B, ElementB>;

  // The prefetch copy is different from the main copy here we use the subgroup collectively to load the data
  using XE_Prefetch_A = decltype(cute::detail::prefetch_selector<PrefetchATileSize, ElementA, InternalStrideA, SubgroupSize>(
      make_tensor(make_gmem_ptr(static_cast<ElementA const *>(nullptr)), make_layout(make_shape(0, 0, 0), InternalStrideA{}))));
  using XE_Prefetch_B = decltype(cute::detail::prefetch_selector<PrefetchBTileSize, ElementB, InternalStrideB, SubgroupSize>(
      make_tensor(make_gmem_ptr(static_cast<ElementB const *>(nullptr)), make_layout(make_shape(0, 0, 0), InternalStrideB{}))));

  using TensorMKL = decltype(make_tensor(make_gmem_ptr(static_cast<ElementA const*>(nullptr)), make_shape(0,0,0), InternalStrideA{}));   //(m, k)
  using TensorNKL = decltype(make_tensor(make_gmem_ptr(static_cast<ElementB const*>(nullptr)), make_shape(0,0,0), InternalStrideB{}));   //(n, k)

  using CopyThreadShape = Shape<_1, Int<SubgroupSize>>;
  using Copy_A = decltype(make_tiled_copy(atom_load_A{},
                                   Layout<CopyThreadShape>{},
                                   make_layout(shape_div(typename traits_load_A::BlockShape{}, CopyThreadShape{}))));
          
  using Copy_B = decltype(make_tiled_copy(atom_load_B{},
                                   Layout<CopyThreadShape>{},
                                   make_layout(shape_div(typename traits_load_B::BlockShape{}, CopyThreadShape{}))));
 
  // Host side kernel arguments
  struct Arguments {
    ElementA const** ptr_A;
    StrideA dA;
    ElementB const** ptr_B;
    StrideB dB;
  };

  struct Params {
    Copy_A copy_A;
    Copy_B copy_B;
    TensorMKL mA;
    TensorNKL mB;
    ElementA const** ptr_A;
    StrideA dA;
    ElementB const** ptr_B;
    StrideB dB;
  };

  //
  // Methods
  //

  CollectiveMma() = default;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    (void) workspace;

    // Batches/Groups are managed by using appropriate pointers to input matrices
    const int32_t mock_L = 1;
    ElementA const* ptr_A_first_batch = reinterpret_cast<ElementA const*>(args.ptr_A);
    ElementB const* ptr_B_first_batch = reinterpret_cast<ElementB const*>(args.ptr_B);

    auto problem_shape_MNK = repeat_like(typename ProblemShape::UnderlyingProblemShape{}, int32_t(1));;
    auto init_M = get<0>(problem_shape_MNK);
    auto init_N = get<1>(problem_shape_MNK);
    auto init_K = get<2>(problem_shape_MNK);

    TensorMKL mA_mkl = make_tensor(make_gmem_ptr(ptr_A_first_batch), make_layout(make_shape(init_M,init_K,mock_L), InternalStrideA{}/*args.dA*/));
    TensorNKL mB_nkl = make_tensor(make_gmem_ptr(ptr_B_first_batch), make_layout(make_shape(init_N,init_K,mock_L), InternalStrideB{}/*args.dB*/));

    auto tiled_copy_a = make_tiled_copy(atom_load_A{}.with(mA_mkl),
                                   Layout<CopyThreadShape>{},
                                   make_layout(shape_div(typename traits_load_A::BlockShape{}, CopyThreadShape{})));
    auto tiled_copy_b = make_tiled_copy(atom_load_B{}.with(mB_nkl),
                                   Layout<CopyThreadShape>{},
                                   make_layout(shape_div(typename traits_load_B::BlockShape{}, CopyThreadShape{})));


    return Params{
      tiled_copy_a,
      tiled_copy_b,
      mA_mkl,
      mB_nkl,
      args.ptr_A,
      args.dA,
      args.ptr_B,
      args.dB
    };
  }

  /// Perform a subgroup-scoped matrix multiply-accumulate
  template <class FrgTensorD, class TensorA, class TensorB, class FrgTensorC, class KTileIterator,
            class BlkCoord, class LoadTensors>
  CUTLASS_DEVICE void operator()(FrgTensorD &accum, TensorA gA, TensorB gB, FrgTensorC const &src_accum,
                                 KTileIterator k_tile_iter, int const& k_tile_count,
                                 BlkCoord const &blk_coord, int const &K_start, int const& thread_idx, char *smem_buf,
                                 Params const &mainloop, LoadTensors const& load_tensors) {
    static_assert(is_rmem<FrgTensorD>::value, "D tensor must be rmem resident.");
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");

    (void)thread_idx;
    (void)smem_buf;
    
    auto thr_copy_A = mainloop.copy_A.get_slice(thread_idx);
    auto thr_copy_B = mainloop.copy_B.get_slice(thread_idx);

    // Instantiate the MMA object and get thread slice
    TiledMma tiled_mma;
    // TODO(Codeplay): see if we can make this nicer
    // To make all work items in a subgroup have the same global tensors pass in the index of work item 0 in each subgroup
    auto sg = syclcompat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * DispatchPolicy::SubgroupSize;
    auto thr_mma = tiled_mma.get_slice(first_thread_in_sg_idx);

    // Partition
    Tensor tCgA = thr_mma.partition_A(gA);
    Tensor tCgB = thr_mma.partition_B(gB);

    Tensor tCrA = make_tensor<ElementA>(mainloop.copy_A.make_fragment_layout(tCgA(_,_,_,0).shape()));
    Tensor tCrB = make_tensor<ElementB>(mainloop.copy_B.make_fragment_layout(tCgB(_,_,_,0).shape()));
  
    // Retile registers for copies
    Tensor tArA = thr_copy_A.retile_D(tCrA);
    Tensor tBrB = thr_copy_B.retile_D(tCrB);
    
    // Retile global tile for copies
    Tensor tAgA = thr_copy_A.retile_S(tCgA);
    Tensor tBgB = thr_copy_B.retile_S(tCgB);


#if CUTLASS_ENABLE_DEBUG_PRINTS
    if (cutlass::thread(LOG_THREAD, LOG_GROUP)) {
        print("======================= A: \n");
        print("  gA : "); print(gA); print("\n");
        print("tCgA : "); print(tCgA); print("\n");
        print("tAgA : "); print(tAgA); print("\n");

        print("=====================  B :\n");
        print("  gB : "); print(gB); print("\n");
        print("tCgB : "); print(tCgB); print("\n");
        print("tBgB : "); print(tBgB); print("\n");

        print("=====================  Config: \n");
        print("  threads per workgroup : "); print(MaxThreadsPerBlock); print("\n");
        print("  SubgroupTileShape : "); print(SubgroupTileShape{}); print("\n");

        print(" PrefetchAThrShape :    ");print(PrefetchAThrShape{});print("\n");
        print(" PrefetchBThrShape :    ");print(PrefetchBThrShape{});print("\n");
        print(" PrefetchATileSize :    ");print(PrefetchATileSize{});print("\n");
        print(" PrefetchBTileSize :    ");print(PrefetchBTileSize{});print("\n");
      }
#endif

    //
    // Mainloop
    //
    auto [m_idx, n_idx, k_idx, l_idx] = blk_coord;

    auto m_sg = get_sub_group_id() / ATOM_N;
    auto n_sg = get_sub_group_id() % ATOM_N;

    const int m_coord = m_idx * BLK_M + (get_sub_group_id() / ATOM_N) * SG_M;
    const int n_coord = n_idx * BLK_N + (get_sub_group_id() % ATOM_N) * SG_N;
    const int l_coord = l_idx;

    Tensor block2d_copy_iter_a = mainloop.copy_A.get_pvc_tensor(make_coord(m_coord, 0, l_coord), tArA.shape());
    auto copy_iter_a = append_pvc_tensor<1>(block2d_copy_iter_a, k_tile_count, BLK_K);

    Tensor block2d_copy_iter_b = mainloop.copy_B.get_pvc_tensor(make_coord(n_coord, 0, l_coord), tBrB.shape());
    auto copy_iter_b = append_pvc_tensor<1>(block2d_copy_iter_b, k_tile_count, BLK_K);
    const auto k_start_idx = crd2idx((*k_tile_iter), make_shape(K_start));

    auto sub_group_id = get_sub_group_id();
    // 0/Hight/vertical
    const int prefetch_a_coord_0 =  is_A_transposed ? k_start_idx * BLK_K : m_idx * BLK_M;
    // 1/Width/Horisontal
    const int prefetch_a_coord_1 =  is_A_transposed ? m_idx * BLK_M : k_start_idx * BLK_K;
    constexpr int ld_a = is_A_transposed ? 0 : 1;

    auto tiled_prefetch_a = XE_Prefetch_A{mainloop.mA};
    auto tiled_prefetch_b = XE_Prefetch_B{mainloop.mB};

    auto prefetch_a_coordinate = make_coord(prefetch_a_coord_0 + (sub_group_id / get<1>(PrefetchAThrShape{})) * get<0>(PrefetchATileSize{}),
                                            prefetch_a_coord_1 + (sub_group_id % get<1>(PrefetchAThrShape{})) * get<1>(PrefetchATileSize{}),
                                            l_coord);        
    Tensor block2d_prefetch_iter_a = tiled_prefetch_a.get_pvc_tensor(prefetch_a_coordinate, make_shape(_1{}, _1{}, _1{}));
     // no transpose prefetch size(8x32), prefetch shape is (32x1)
    Tensor prefetch_iter_a = append_pvc_tensor<ld_a>(block2d_prefetch_iter_a, k_tile_count,
                                                     (get<ld_a>(PrefetchAThrShape{}) * get<ld_a>(PrefetchATileSize{})));
     // 0/Hight/vertical/ 
    const int prefetch_b_coord_0 =  is_B_transposed ? n_idx * BLK_N: k_start_idx * BLK_K;
    // 1/Width/Horisontal 
    const int prefetch_b_coord_1 =  is_B_transposed ? k_start_idx * BLK_K : n_idx * BLK_N;
    constexpr int ld_b = is_B_transposed ? 1 : 0;
    auto prefetch_b_coordinate = make_coord(prefetch_b_coord_0 + (sub_group_id / get<1>(PrefetchBThrShape{})) * get<0>(PrefetchBTileSize{}),
                                            prefetch_b_coord_1 + (sub_group_id % get<1>(PrefetchBThrShape{})) * get<1>(PrefetchBTileSize{}),
                                            l_coord);
    Tensor block2d_prefetch_iter_b = tiled_prefetch_b.get_pvc_tensor(prefetch_b_coordinate, make_shape(_1{}, _1{}, _1{}));
    // no transpose prefetch size(8x32), prefetch shape is (4x8)
    Tensor prefetch_iter_b = append_pvc_tensor<ld_b>(block2d_prefetch_iter_b, k_tile_count,
                                                     (get<ld_b>(PrefetchBThrShape{}) * get<ld_b>(PrefetchBTileSize{})));

    constexpr int barrier_scope = 2;
    int prefetch_k = 0;

    CUTLASS_PRAGMA_UNROLL
    for (; prefetch_k < DispatchPolicy::Stages; prefetch_k++) {
      prefetch(tiled_prefetch_a.with(get<0>(load_tensors)), prefetch_iter_a(_, _, _, prefetch_k));
      prefetch(tiled_prefetch_b.with(get<1>(load_tensors)), prefetch_iter_b(_, _, _, prefetch_k));
    }

    CUTLASS_PRAGMA_UNROLL
    for (int k_tile = k_start_idx; k_tile < k_tile_count + k_start_idx; k_tile++, prefetch_k++) {
      barrier_arrive(barrier_scope);
      // Copy gmem to rmem for the first k_tile
      copy(mainloop.copy_A.with(get<0>(load_tensors)), tAgA(_,_,_,k_tile), tArA);
      copy(mainloop.copy_B.with(get<1>(load_tensors)), tBgB(_,_,_,k_tile), tBrB);

      if (prefetch_k < k_tile_count) {
        prefetch(tiled_prefetch_a.with(get<0>(load_tensors)), prefetch_iter_a(_, _, _, prefetch_k));
        prefetch(tiled_prefetch_b.with(get<1>(load_tensors)), prefetch_iter_b(_, _, _, prefetch_k));
      }

      cute::gemm(tiled_mma, tCrA, tCrB, accum);
      barrier_wait(barrier_scope);
    }
  }

  template <typename ProblemShape_MNKL>
  CUTLASS_DEVICE auto update_tensor_shape_stride(
    Params const& mainloop_params,
    int32_t const& next_group,
    ProblemShape_MNKL const& problem_shape_mnkl) {
      const int32_t M = get<0>(problem_shape_mnkl);
      const int32_t N = get<1>(problem_shape_mnkl);
      const int32_t K = get<2>(problem_shape_mnkl);

      ElementA const* ptr_A_curr_batch = reinterpret_cast<ElementA const*>(mainloop_params.ptr_A[next_group]);
      ElementB const* ptr_B_curr_batch = reinterpret_cast<ElementB const*>(mainloop_params.ptr_B[next_group]);

      Tensor mA = make_tensor(make_gmem_ptr(ptr_A_curr_batch), make_shape(M, K,(int32_t)1), mainloop_params.dA[next_group]);
      Tensor mB = make_tensor(make_gmem_ptr(ptr_B_curr_batch), make_shape(N, K,(int32_t)1), mainloop_params.dB[next_group]);

      return cute::make_tuple(mA, mB);
    }
};

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
