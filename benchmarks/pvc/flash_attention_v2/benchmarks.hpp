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

#include "benchmark_runner.hpp"
#include "fmha_configuration.hpp"

using TiledMma1 = TiledMMA<MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>, Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>,
                            Tile<Layout<Shape<_8, _8, _2>, Stride<_1, _16, _8>>,
                                Layout<Shape<_16, _1, _4>, Stride<_1, _64, _16>>,
                                _64>>;

using TiledMma2 = TiledMMA<MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>, Layout<Shape<_8, _2, _1>, Stride<_2, _1, _1>>,
                            Tile<Layout<Shape<_8, _8, _2>, Stride<_1, _16, _8>>,
                                Layout<Shape<_16, _2, _4>, Stride<_1, _64, _16>>,
                                _64>>;

using PvcFMHABF16BF16FP32_RCR_1 = cutlass::flash_attention::FMHAConfig<
        cutlass::layout::RowMajor,
        cutlass::layout::ColumnMajor,
        cutlass::layout::RowMajor,
        cutlass::layout::RowMajor, 
        true, Shape<_128, _64, _64>,
        TiledMma1>;

using PvcFMHABF16BF16FP32_RCR_2 = cutlass::flash_attention::FMHAConfig<
        cutlass::layout::RowMajor,
        cutlass::layout::ColumnMajor,
        cutlass::layout::RowMajor,
        cutlass::layout::RowMajor,
        false, Shape<_128, _64, _64>,
        TiledMma1>;

using PvcFMHABF16BF16FP32_RCR_3 = cutlass::flash_attention::FMHAConfig<
        cutlass::layout::RowMajor,
        cutlass::layout::ColumnMajor,
        cutlass::layout::RowMajor,
        cutlass::layout::RowMajor, 
        true, Shape<_128, _128, _64>,
        TiledMma2>;

using PvcFMHABF16BF16FP32_RCR_4 = cutlass::flash_attention::FMHAConfig<
        cutlass::layout::RowMajor,
        cutlass::layout::ColumnMajor,
        cutlass::layout::RowMajor,
        cutlass::layout::RowMajor, 
        false, Shape<_128, _128, _64>,
        TiledMma2>;

CUTLASS_CREATE_FMHA_BENCHMARK(PvcFMHABF16BF16FP32_RCR_1);
CUTLASS_CREATE_FMHA_BENCHMARK(PvcFMHABF16BF16FP32_RCR_2);
CUTLASS_CREATE_FMHA_BENCHMARK(PvcFMHABF16BF16FP32_RCR_3);
CUTLASS_CREATE_FMHA_BENCHMARK(PvcFMHABF16BF16FP32_RCR_4);

static void register_benchmarks() {
  CUTLASS_FMHA_BENCHMARK(PvcFMHABF16BF16FP32_RCR_1);
  CUTLASS_FMHA_BENCHMARK(PvcFMHABF16BF16FP32_RCR_2);
  CUTLASS_FMHA_BENCHMARK(PvcFMHABF16BF16FP32_RCR_3);
  CUTLASS_FMHA_BENCHMARK(PvcFMHABF16BF16FP32_RCR_4);
}
