/***************************************************************************************************
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#define CUTLASS_DEBUG_TRACE_LEVEL 1

#include "cutlass_unit_test.h"

#include <cutlass/trace.h>

#include <iostream>

#include <cute/tensor.hpp>

using namespace cute;


template <class LayoutA, class LayoutB>
void
test_distribute(LayoutA const& layoutA,
                LayoutB const& layoutB)
{
  auto layoutR = domain_distribute(shape(layoutA), shape(layoutB));

  CUTLASS_TRACE_HOST("test_distribute()");
  CUTLASS_TRACE_HOST(layoutA << "  <->  " << layoutB);
  CUTLASS_TRACE_HOST("  =>  ");
  CUTLASS_TRACE_HOST(layoutR);

  // Test that layout B is softly compatible with layout R
  EXPECT_TRUE(softly_compatible(layoutB, layoutR));

  // Post-condition on the codomain of the distribute
  for (int i = 0; i < size(layoutR); ++i) {
    for (int j = i+1; j < size(layoutR); ++j) {
      EXPECT_TRUE(layoutR(i) < layoutR(j));     // Surjective and Ordered
    }
  }
}


TEST(CuTe_core, Distribute)
{
  CUTLASS_TRACE_HOST("-------------------------------");
  CUTLASS_TRACE_HOST("DOMAIN DISTRIBUTE"              );
  CUTLASS_TRACE_HOST("-------------------------------");

  {
    auto shape_a = Shape<Shape<_64,_3>,Shape<_8,_8>>{};
    auto shape_b = _128{};

    test_distribute(shape_a, shape_b);
  }

  {
    auto shape_a = Shape<Int<192>,Shape<_8,_8>>{};
    auto shape_b = _128{};

    test_distribute(shape_a, shape_b);
  }

  {
    auto shape_a = Shape<Shape<_64,_3>,Shape<_8,_8>>{};
    auto shape_b = _128{} * _8{};

    test_distribute(shape_a, shape_b);
  }

  {
    auto shape_a = Shape<Int<192>,Shape<_8,_8>>{};
    auto shape_b = _128{} * _8{};

    test_distribute(shape_a, shape_b);
  }

  {
    auto shape_a = Shape<Shape<_64,_3>>{};
    auto shape_b = _128{};

    test_distribute(shape_a, shape_b);
  }
}
