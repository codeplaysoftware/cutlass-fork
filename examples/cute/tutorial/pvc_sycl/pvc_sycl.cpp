/*
// Copyright (c) 2019-2024 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <sycl.hpp>

#include <algorithm>
#include <chrono>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "../gemm_validation.hpp"
#include "pvc_prefetch.hpp"
#include <cute/numeric/arithmetic_tuple.hpp>
#include <cute/tensor.hpp>

using test_clock = std::chrono::high_resolution_clock;

using namespace cute;

using dtype_a = bfloat16_t;
using dtype_b = bfloat16_t;
using dtype_c = float;
using dtype_acc = float;

size_t testIterations = 50;
dtype_acc threshold = 0.01f;

#define B_VNNI

#define WARMUP_ITERATIONS 0

#define PREFETCH_DISTANCE 1

#define split_barrier_arrive() __builtin_IB_work_group_barrier_arrive(0)
#define split_barrier_wait() __builtin_IB_work_group_barrier_wait(0)

template <typename T>
static void init_matrix(T *M, size_t numRows, size_t numCols) {
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_real_distribution<float> dist(-1.0, 1.0);
  for (size_t r = 0; r < numRows; r++) {
    for (size_t c = 0; c < numCols; c++) {
      M[r * numCols + c] = bfloat16_t(dist(rng));
    }
  }
}

template <typename T>
static void vnni_matrix(T *dst, const T *src, size_t numRows, size_t numCols,
                        size_t factor) {
  for (size_t r = 0; r < numRows / factor; r++) {
    for (size_t c = 0; c < numCols; c++) {
      for (size_t k = 0; k < factor; k++) {
        dst[r * numCols * factor + c * factor + k] =
            src[(r * factor + k) * numCols + c];
      }
    }
  }
}

template <typename T>
void check_results(size_t M, size_t N, const T *C, const T *C_ref) {
  float err = 0.f;
  size_t error_cnt = 0;
  for (size_t m = 0; m < M; m++) {
    for (size_t n = 0; n < N; n++) {
      auto index = m * N + n;
      auto localErr = std::fabs(C[index] - C_ref[index]) /
                      std::max(std::fabs(C[index]), std::fabs(C_ref[index]));
      err = std::max(localErr, err);
      if (localErr >= threshold) {
        error_cnt++;
        // std::cerr << "Error at m = " << m << ", n = " << n << ": (local
        // error"
        //           << localErr << "): Wanted " << C_ref[index] << ", got "
        //          << C[index] << std::endl;
        // return;
      }
    }
  }

  auto pass_rate = (1.f - ((float)error_cnt / (M * N))) * 100; // %

  std::cout << "\n\n==== Pass rate is: " << pass_rate << "% !!!\n" << std::endl;
}

inline size_t time_event(sycl::event &e) {
  // get start and end times
  cl_ulong start_time = e.template get_profiling_info<
      sycl::info::event_profiling::command_start>();

  cl_ulong end_time =
      e.template get_profiling_info<sycl::info::event_profiling::command_end>();

  // return the delta
  return static_cast<size_t>(end_time - start_time);
}

int main(int argc, char **argv) {
  auto queue = sycl::queue{{sycl::property::queue::enable_profiling()}};
  auto context = queue.get_info<sycl::info::queue::context>();
  auto device = queue.get_info<sycl::info::queue::device>();

  static constexpr auto M = 4;
  static constexpr auto N = 7168;
  static constexpr auto K = 4096;

  static constexpr auto wg_tile_m = 8;
  static constexpr auto wg_tile_n = 512;

  static constexpr auto sg_tile_m = 8;
  static constexpr auto sg_tile_n = 32;

  static constexpr auto item_tile_m = 8;
  static constexpr auto item_tile_n = 2;

  static constexpr auto sg_per_wg_m = wg_tile_m / sg_tile_m;
  static constexpr auto sg_per_wg_n = wg_tile_n / sg_tile_n;

  static constexpr auto tM = 8;
  static constexpr auto tK = 16;
  static constexpr auto tN = 16;

  static constexpr auto MM = (sg_tile_m + tM - 1) / tM;
  static constexpr auto KK = 2;
  static constexpr auto NN = (sg_tile_n + tN - 1) / tN;

  dtype_a *A_host = (dtype_a *)syclcompat::malloc_host(sizeof(dtype_a) * M * K);
  dtype_b *B_host = (dtype_b *)syclcompat::malloc_host(sizeof(dtype_b) * N * K);
  dtype_c *C_host = (dtype_c *)syclcompat::malloc_host(sizeof(dtype_c) * M * N);

  dtype_a *A_dev =
      (dtype_a *)sycl::malloc_device(sizeof(dtype_a) * M * K, device, context);
  dtype_b *B_dev =
      (dtype_b *)sycl::malloc_device(sizeof(dtype_b) * N * K, device, context);
  dtype_acc *C_dev = (dtype_acc *)sycl::malloc_device(sizeof(dtype_c) * M * N,
                                                      device, context);

  printf("Initializing source matrices...\n");
  init_matrix(A_host, M, K);
  init_matrix(B_host, K, N);

  dtype_b *Bvnni_host =
      (dtype_b *)syclcompat::malloc_host(sizeof(dtype_b) * N * K);
  vnni_matrix(Bvnni_host, B_host, K, N, 2);

  queue.memcpy(A_dev, A_host, sizeof(dtype_a) * M * K);
  queue.memcpy(B_dev, Bvnni_host, sizeof(dtype_b) * N * K);
  queue.memcpy(C_dev, C_host, sizeof(dtype_c) * M * N);

  printf("Computing reference...\n");
  dtype_acc *C_ref_host =
      (dtype_acc *)syclcompat::malloc_host(sizeof(dtype_acc) * M * N);

  get_gemm_gold<dtype_a, dtype_b, dtype_acc>(
      M, N, K, mem_layout::row_major, mem_layout::row_major, (dtype_a *)A_host,
      (dtype_b *)B_host, (dtype_c *)C_ref_host);

  printf("Running gemm tests, MKN: (%d, %d, %d)...\n", M, K, N);

  const uint32_t total_iterations = WARMUP_ITERATIONS + testIterations;

  std::vector<float> event_times(total_iterations);

  sycl::range<2> group_range{(N + wg_tile_n - 1) / wg_tile_n,
                             (M + wg_tile_m - 1) / wg_tile_m};
  sycl::range<2> local_range{(wg_tile_n + item_tile_n - 1) / item_tile_n,
                             (wg_tile_m + item_tile_m - 1) / item_tile_m};
  sycl::nd_range<2> nd_range(group_range * local_range, local_range);

  for (uint32_t test = 0; test < total_iterations; test++) {
    sycl::event ev;
    ev = queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
          nd_range, [=](sycl::nd_item<2> id) [[sycl::reqd_sub_group_size(16)]] {
            const int m = id.get_group(1) * wg_tile_m +
                          (get_sub_group_id() / sg_per_wg_n) * sg_tile_m;
            const int n = id.get_group(0) * wg_tile_n +
                          (get_sub_group_id() % sg_per_wg_n) * sg_tile_n;

            Tensor tAr = make_tensor<ushort>(Shape<Int<8 * KK * MM>, Int<1>>{});
            Tensor tBr = make_tensor<ushort>(Shape<Int<16 * KK>, Int<NN>>{});
            Tensor tCr = make_tensor<dtype_acc>(Shape<_8, Int<MM>, Int<NN>>{});

            auto A_copy = make_xe_2d_A_copy(
                make_tensor(make_gmem_ptr(A_dev), make_shape(M, K)));
            auto B_copy = make_xe_2d_B_copy(
                make_tensor(make_gmem_ptr(B_dev), make_shape(K, N)));
            auto C_copy = make_xe_2d_copy(
                make_tensor(make_gmem_ptr(C_dev), make_shape(M, N)));
            // TODO: - decide on how to deal with vector types
            //       - create layouts with tiling/partitioning

            Tensor tAi = make_tensor(
                make_inttuple_iter(m, 0),
                make_layout(make_shape(_1{}, _1{}, K),
                            make_stride(_1{}, MM * tM * E<0>{}, E<1>{})));
            Tensor tBi = make_tensor(
                make_inttuple_iter(0, n),
                make_layout(make_shape(_1{}, K, Int<NN>{}),
                            make_stride(_1{}, E<0>{}, tN * E<1>{})));
            Tensor tCi = make_tensor(
                make_inttuple_iter(m, n),
                make_layout(Shape<_1, Int<MM>, Int<NN>>{},
                            make_stride(_1{}, tM * E<0>{}, tN * E<1>{})));
            TiledMMA<MMA_Atom<XE_8x16x16_BF16BF16F32F32_NN>,
                     Layout<Shape<_1, _1, _1>>>
                tiled_mma;

            uint32_t prefetch_k = 0;
#ifdef PREFETCH_DEFAULT
            for (uint32_t p = 0; p < PREFETCH_DISTANCE; p++) {
#ifdef B_VNNI
              HELPER_NAME(btile_block_prefetch_vnni, 4, 4)
              ((ushort *)B_dev, tN, K, N, prefetch_k, n);
#else
                HELPER_NAME(btile_block_prefetch_rowmajor, 4, 4)
                ((ushort *)B_dev, tN, K, N, prefetch_k, n);
#endif
              HELPER_NAME(atile_block_prefetch_rowmajor, 4, 4)
              ((ushort *)A_dev, tM, M, K, m, prefetch_k);
              prefetch_k += tK * KK;
            }
#endif

            for (int k = 0; k < K + tK * KK - 1; k += tK * KK) {
              copy(A_copy, tAi(_, _, k), tAr);
              copy(B_copy, tBi(_, k / KK, _), tBr);

#ifdef PREFETCH_DEFAULT
              for (uint32_t p = 0; p < PREFETCH_DISTANCE; p++) {
#ifdef B_VNNI
                HELPER_NAME(btile_block_prefetch_vnni, 4, 4)
                ((ushort *)B_dev, tN, K, N, prefetch_k, n);
#else
                  HELPER_NAME(btile_block_prefetch_rowmajor, 4, 4)
                  ((ushort *)B_dev, tN, K, N, prefetch_k, n);
#endif
                HELPER_NAME(atile_block_prefetch_rowmajor, 4, 4)
                ((ushort *)A_dev, tM, M, K, m, prefetch_k);
                prefetch_k += tK * KK;
              }
#endif
              auto tAr_view =
                  make_tensor(static_cast<decltype(tAr) &&>(tAr).data(),
                              Shape<_8, Int<MM>, Int<KK>>{});
              auto tBr_view =
                  make_tensor(static_cast<decltype(tBr) &&>(tBr).data(),
                              Shape<_16, Int<KK>, Int<NN>>{});
              for (uint32_t kl = 0; kl < KK; kl++) {
                gemm(tiled_mma, tAr_view(_, _, kl), tBr_view(_, kl, _), tCr);
              }
            }

            copy(C_copy, tCr, tCi);
          });
    });

    ev.wait_and_throw();
    event_times[test] = time_event(ev) / 1e6;
  }

  double average_event_time = 0.f;
  auto best = 999.f;
  for (uint32_t i = WARMUP_ITERATIONS; i < total_iterations; i++) {
    printf("GPU time is %f ms, Tflops is: %f\n", event_times[i],
           2.0 * M * N * K / 1e12 / (event_times[i] / 1e3));
    average_event_time += event_times[i];
    best = min(best, event_times[i]);
  }
  average_event_time /= testIterations;
  printf("Average is %f Tflops, best is %f Tflops\n",
         2.0 * M * N * K / 1e12 / (average_event_time / 1e3),
         2.0 * M * N * K / 1e12 / (best / 1e3));

  printf("Checking results... ");
  fflush(stdout);

  queue.memcpy(C_host, C_dev, M * N * sizeof(dtype_c));
  check_results(M, N, C_host, C_ref_host);

  printf(" done!\n");

  free(A_host, queue);
  free(B_host, queue);
  free(C_host, queue);
  free(Bvnni_host, queue);
  free(C_ref_host, queue);
  free(A_dev, queue);
  free(B_dev, queue);
  free(C_dev, queue);

  return 0;
}
