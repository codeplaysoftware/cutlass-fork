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

#include <cute/atom/copy_atom.hpp>
#include <cute/atom/copy_traits.hpp>

#include <cute/algorithm/prefetch.hpp>
#include <cute/arch/copy_xe.hpp>

namespace cute {

namespace detail
{
  struct MKL_Indicator {};
  struct NKL_Indicator {};

  template <class MNKL_Indicator, class Enable = void>
  struct is_MKL_layout {
    static constexpr bool value = false;
  };

  template <class MNKL_Indicator>
  struct is_MKL_layout<MNKL_Indicator, std::enable_if_t<std::is_same_v<MNKL_Indicator, MKL_Indicator>>> {
    static constexpr bool value = true;
  };

  template <class MNKL_Indicator, class Enable = void>
  struct is_NKL_layout {
    static constexpr bool value = false;
  };

  template <class MNKL_Indicator>
  struct is_NKL_layout<MNKL_Indicator, std::enable_if_t<std::is_same_v<MNKL_Indicator, NKL_Indicator>>> {
    static constexpr bool value = true;
  };

  template <class MNKL_Indicator, class Stride_t>
  struct is_transpose_load{
     static constexpr bool value = (is_MKL_layout<MNKL_Indicator>::value
                                        && std::is_same_v<cutlass::detail::StrideToLayoutTagA_t<Stride_t>, cutlass::layout::ColumnMajor>)
                                   || (is_NKL_layout<MNKL_Indicator>::value
                                        && std::is_same_v<cutlass::detail::StrideToLayoutTagB_t<Stride_t>, cutlass::layout::ColumnMajor>);
  };

  template <class, class Enable = void> constexpr bool has_inst_dtype = false;

  template <class T>
  constexpr bool has_inst_dtype<T, cute::void_t<typename T::inst_dtype>> = true;

  template <class T, class dtype, class Enable = void>
  struct size_of_inst {
    static constexpr auto value = sizeof(dtype);
  };

  template <class T, class dtype>
  struct size_of_inst<T, dtype, std::enable_if_t<has_inst_dtype<T>>> {
    static constexpr auto value = sizeof(typename T::inst_dtype);
  };

  template <uint32_t dim, class Tensor_t>
  static constexpr auto append_pvc_tensor(Tensor_t const &t0, uint32_t shape, uint32_t stride) {
    return make_tensor(make_inttuple_iter(t0.data()), append(t0.layout(), make_layout(shape, E<dim>{} * stride)));
  }

} // namespace detail end

template <class CopyOp,
          class LayoutIndicator = detail::MKL_Indicator,
          class GStride = std::conditional_t<std::is_same_v<LayoutIndicator, detail::MKL_Indicator>,
                                             cute::Stride<int64_t, cute::Int<1>, int64_t>,
                                             cute::Stride<cute::Int<1>, int64_t, int64_t>>>
struct XE_2D_LD_Unpack {
  const void *base_ptr;
  uint32_t width;
  uint32_t height;
  uint32_t pitch;
  
  using Shape_MN = CopyOp::Shape_MN;

  static constexpr bool is_mkl = detail::is_MKL_layout<LayoutIndicator>::value;
  static constexpr bool is_nkl = detail::is_NKL_layout<LayoutIndicator>::value;
  static constexpr bool is_transpose = detail::is_transpose_load<LayoutIndicator, GStride>::value;

  static_assert(is_mkl != is_nkl);

  XE_2D_LD_Unpack(const void *ptr, uint32_t const &y,
                  uint32_t const &x, uint32_t const &p = 0) : base_ptr(ptr) {
    if constexpr (is_nkl) {
      width = is_transpose ? x : y;
      height = is_transpose ? y : x;
      pitch = (p == 0 ? width : p);
    } else {
      width = is_transpose ? y : x;
      height = is_transpose ? x : y;
      pitch = (p == 0 ? width : p);
    }
  }

  template <class TraitsArgs>
  XE_2D_LD_Unpack(TraitsArgs const &traits) : base_ptr(traits.base_ptr),
              width(traits.width), height(traits.height), pitch(traits.pitch) {}

  XE_2D_LD_Unpack() {}

  using Traits_LD_t = Copy_Traits<CopyOp, LayoutIndicator, GStride>;

  template <class TS, class SLayout, class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Traits_LD_t const &traits, Tensor<TS, SLayout> const &src,
              Tensor<TD, DLayout> &dst) {
    static_assert(is_rmem<TD>::value);

    using dtype = typename Tensor<TD, DLayout>::value_type;

    dtype *base_addr = (dtype *)traits.base_ptr;
  
    int x, y;
    auto [coord_0, coord_1, z] = src.data().coord_;
    if constexpr (is_mkl ^ is_transpose) {
      x = coord_1;
      y = coord_0;
    } else {
      x = coord_0;
      y = coord_1;
    }

    static constexpr auto inst_size = detail::size_of_inst<CopyOp, dtype>::value;
 
    CopyOp::copy(base_addr + z * traits.width * traits.height,
                 traits.width * sizeof(dtype), traits.height,
                 traits.pitch * sizeof(dtype),
                 intel::coord_t{(int)(x * sizeof(dtype) / inst_size), y},
                 &*dst.data());
  }

  template <class... CA_Args, class TS, class SLayout>
  CUTE_HOST_DEVICE friend constexpr void
  prefetch(Copy_Atom<Traits_LD_t, CA_Args...> const &atom,
           Tensor<TS, SLayout> const &src) {
    static_assert(detail::has_prefetch<CopyOp>);

    using dtype = typename Copy_Atom<Traits_LD_t, CA_Args...>::ValType;

    dtype *base_addr = (dtype *)atom.base_ptr;

    auto [m, n, l] = src.data().coord_;

    CopyOp::PREFETCH::copy((void *)(base_addr + l * atom.width * atom.height),
                           atom.width * sizeof(dtype), atom.height,
                           atom.pitch * sizeof(dtype),
                           intel::coord_t{(int)n, (int)m});
  }

  template <class GShape>
  CUTE_HOST_DEVICE constexpr auto get_pvc_tensor(int m_coord, int n_coord, int l_coord,
                                                 GShape const &shape) const {

    auto R = rank(GShape{});
    static_assert(R == 3, "mismatch rank");

    using basis_t =  make_seq<rank(typename CopyOp::Shape_MN{})>;
    using rvs_basis_t = decltype(reverse(basis_t{}));
    using use_basis_t = std::conditional_t<is_mkl, basis_t, rvs_basis_t>;

    if constexpr (is_mkl) {
      auto new_shape = cute::tuple_cat(make_shape(_1{}), take<R - 2, R>(shape));
      auto new_stride = cute::tuple_cat(make_stride(_1{}), transform(basis_t{}, typename CopyOp::Shape_MN{},
                                                                    [&](auto i, auto s){
                                                                        return E<i>{} * s;
                                                                    }));
      return make_tensor(make_inttuple_iter(make_coord(m_coord, n_coord, l_coord)),
                         make_layout(new_shape, new_stride));

    } else {
      auto new_shape = cute::tuple_cat(make_shape(_1{}), (take<R - 2, R>(shape)));
      auto new_stride = cute::tuple_cat(make_stride(_1{}), transform(basis_t{}, reverse(typename CopyOp::Shape_MN{}),
                                                                    [&](auto i, auto s){
                                                                        return E<i>{} * s;
                                                                    }));
      return make_tensor(make_inttuple_iter(make_coord(m_coord, n_coord, l_coord)),
                         make_layout(new_shape, new_stride));
    }
  }

  template <class T1, class T2, class... TraitsArgs>
  static constexpr auto with(T1 && arg1, T2 && arg2, TraitsArgs&&... args) {
      return Traits_LD_t{arg1, arg2, args...};
  }
};

template <class CopyOp, class... ArgTs> struct XE_2D_ST_Unpack {
  const void *base_ptr;
  uint32_t width;
  uint32_t height;
  uint32_t pitch;

  XE_2D_ST_Unpack(const void *ptr, uint32_t const &w,
                  uint32_t const &h, uint32_t const &p)
      : base_ptr(ptr), width(w), height(h), pitch(p) {}

  template <class TraitsArgs>
  XE_2D_ST_Unpack(TraitsArgs const &traits)  : base_ptr(traits.base_ptr),
                  width(traits.width), height(traits.height), pitch(traits.pitch) {}

  XE_2D_ST_Unpack() {}

  using Traits_ST_t = Copy_Traits<CopyOp, ArgTs...>;

  template <class TS, class SLayout, class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Traits_ST_t const &traits, Tensor<TS, SLayout> const &src,
              Tensor<TD, DLayout> &dst) {
    static_assert(is_rmem<TS>::value);

    using dtype = typename Tensor<TS, SLayout>::value_type;

    dtype *base_addr = (dtype *)traits.base_ptr;

    auto [m, n, l] = dst.data().coord_;

    CopyOp::copy(base_addr + l * traits.width * traits.height,
                 (int)(traits.width * sizeof(dtype)), (int)(traits.height),
                 (int)(traits.pitch * sizeof(dtype)),
                 intel::coord_t{(int)n, (int)m}, &*src.data());
  }

  template <class GCoord, class GShape, class GStride, class Basis = decltype(make_seq<rank(GStride{})>{})>
  CUTE_HOST_DEVICE constexpr auto get_pvc_tensor(GCoord const &coord,
                                                 GShape const &shape,
                                                 GStride const &stride, 
                                                 Basis const & basis = {}) const {

    auto R = rank(GShape{});
    static_assert(R == 3 || R == 4, "mismatch rank");
    auto t_shape = cute::tuple_cat(make_shape(_1{}), take<1, R>(shape));
    auto t_stride =  cute::tuple_cat(make_stride(_1{}), transform(basis, stride, [&](auto i, auto s){
        return E<i>{} * s;
    }));
    return make_tensor(make_inttuple_iter(coord),
                       make_layout(t_shape, t_stride));
  }

  template <class T1, class T2, class... TraitsArgs>
  static constexpr auto with(T1 && arg1, T2 && arg2, TraitsArgs&&... args) {
      return Traits_ST_t{arg1, arg2, args...};
  }
};

// clang-format off

template <class... args_t>
struct Copy_Traits<XE_2D_U8x1x32_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U8x1x32_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_8>,
                           Stride< _0,_1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,_16>,
                           Stride<_16, _1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_LD_Unpack<XE_2D_U8x1x32_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x2x32_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U8x2x32_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_8>,
                           Stride< _0,_1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _2>>,
                           Stride<_16,Stride< _1,_256>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_LD_Unpack<XE_2D_U8x2x32_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x4x32_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U8x4x32_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_8>,
                           Stride< _0,_1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _4>>,
                           Stride<_16,Stride< _1,_256>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_LD_Unpack<XE_2D_U8x4x32_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x8x32_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U8x8x32_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_8>,
                           Stride< _0,_1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _8>>,
                           Stride<_16,Stride< _1,_256>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_LD_Unpack<XE_2D_U8x8x32_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x16x32_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U8x16x32_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_8>,
                           Stride< _0,_1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16, _16>>,
                           Stride<_16,Stride< _1,_256>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_LD_Unpack<XE_2D_U8x16x32_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x32x32_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U8x32x32_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_8>,
                           Stride< _0,_1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16, _32>>,
                           Stride<_16,Stride< _1,_256>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_LD_Unpack<XE_2D_U8x32x32_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x1x64_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U8x1x64_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_8>,
                           Stride< _0,_1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _2>>,
                           Stride<_16,Stride< _1,_256>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_LD_Unpack<XE_2D_U8x1x64_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x1x64_LD_N::PREFETCH, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U8x1x64_LD_N, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,Shape <_8,_2,  _2>>,
                           Stride<_16,Stride<_1,_8,_256>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_8,_2,  _2>>,
                           Stride<_16,Stride<_1,_8,_256>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
  using CopyInternalType = ushort;
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x2x64_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U8x2x64_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_8>,
                           Stride< _0,_1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _2,  _2>>,
                           Stride<_16,Stride< _1,_256,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_LD_Unpack<XE_2D_U8x2x64_LD_N, args_t...>(args...) {}
};


template <class... args_t>
struct Copy_Traits<XE_2D_U8x2x64_LD_N::PREFETCH, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U8x2x64_LD_N, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,Shape <_8,_2,  _2,  _2>>,
                           Stride<_16,Stride<_1,_8,_256,_512>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_8,_2,  _2,  _2>>,
                           Stride<_16,Stride<_1,_8,_256,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
  using CopyInternalType = ushort;
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x4x64_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U8x4x64_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_8>,
                           Stride< _0,_1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _2,  _4>>,
                           Stride<_16,Stride< _1,_256,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_LD_Unpack<XE_2D_U8x4x64_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x4x64_LD_N::PREFETCH, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U8x4x64_LD_N, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,Shape <_8,_2,  _2,  _4>>,
                           Stride<_16,Stride<_1,_8,_256,_512>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_8,_2,  _2,  _4>>,
                           Stride<_16,Stride<_1,_8,_256,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
  using CopyInternalType = ushort;
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x8x64_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U8x8x64_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_8>,
                           Stride< _0,_1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _2,  _8>>,
                           Stride<_16,Stride< _1,_256,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_LD_Unpack<XE_2D_U8x8x64_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x8x64_LD_N::PREFETCH, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U8x8x64_LD_N::PREFETCH, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,Shape <_8,_2,  _2,  _8>>,
                           Stride<_16,Stride<_1,_8,_256,_512>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_8,_2,  _2,  _8>>,
                           Stride<_16,Stride<_1,_8,_256,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
  using CopyInternalType = ushort;
};


template <class... args_t>
struct Copy_Traits<XE_2D_U8x16x64_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U8x16x64_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_8>,
                           Stride< _0,_1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _2, _16>>,
                           Stride<_16,Stride< _1,_256,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_LD_Unpack<XE_2D_U8x16x64_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x16x64_LD_N::PREFETCH, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U8x16x64_LD_N::PREFETCH, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,Shape <_16,  _2, _16>>,
                           Stride<_16,Stride< _1,_256,_512>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _2, _16>>,
                           Stride<_16,Stride< _1,_256,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x32x64_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U8x32x64_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_8>,
                           Stride< _0,_1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _2, _32>>,
                           Stride<_16,Stride< _1,_256,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_LD_Unpack<XE_2D_U8x32x64_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x32x64_LD_N::PREFETCH, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U8x32x64_LD_N::PREFETCH, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,Shape <_16,  _2, _32>>,
                           Stride<_16,Stride< _1,_256,_512>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _2, _32>>,
                           Stride<_16,Stride< _1,_256,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x1x16_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x1x16_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_16>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,_16>,
                           Stride<_16, _1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_LD_Unpack<XE_2D_U16x1x16_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x2x16_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x2x16_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_16>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _2>>,
                           Stride<_16,Stride< _1,_256>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_LD_Unpack<XE_2D_U16x2x16_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x4x16_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x4x16_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_16>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _4>>,
                           Stride<_16,Stride< _1,_256>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_LD_Unpack<XE_2D_U16x4x16_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x8x16_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x8x16_LD_N, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_16>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _8>>,
                           Stride<_16,Stride< _1,_256>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U16x8x16_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x8x16_LD_N::PREFETCH, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x8x16_LD_N, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,Shape <_16,  _8>>,
                           Stride<_16,Stride< _1,_256>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _8>>,
                           Stride<_16,Stride< _1,_256>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x16x16_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x16x16_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_16>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16, _16>>,
                           Stride<_16,Stride< _1,_256>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U16x16x16_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x16x16_LD_N::PREFETCH, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x16x16_LD_N, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,Shape <_16, _16>>,
                           Stride<_16,Stride< _1,_256>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16, _16>>,
                           Stride<_16,Stride< _1,_256>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x32x16_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x32x16_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_16>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16, _32>>,
                           Stride<_16,Stride< _1,_256>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U16x32x16_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x32x16_LD_N::PREFETCH, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x32x16_LD_N, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,Shape <_16, _32>>,
                           Stride<_16,Stride< _1,_256>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16, _32>>,
                           Stride<_16,Stride< _1,_256>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x1x32_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x1x32_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_16>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _2>>,
                           Stride<_16,Stride< _1,_256>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U16x1x32_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x1x32_LD_N::PREFETCH, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x1x32_LD_N, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,Shape <_16,  _2>>,
                           Stride<_16,Stride< _1,_256>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _2>>,
                           Stride<_16,Stride< _1,_256>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x2x32_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x2x32_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_16>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _2,  _2>>,
                           Stride<_16,Stride< _1,_256,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U16x2x32_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x2x32_LD_N::PREFETCH, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x2x32_LD_N, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,Shape <_16,  _2,  _2>>,
                           Stride<_16,Stride< _1,_256,_512>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _2,  _2>>,
                           Stride<_16,Stride< _1,_256,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x4x32_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x4x32_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_16>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _2,  _4>>,
                           Stride<_16,Stride< _1,_256,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U16x4x32_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x4x32_LD_N::PREFETCH, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x4x32_LD_N, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,Shape <_16,  _2,  _4>>,
                           Stride<_16,Stride< _1,_256,_512>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _2,  _4>>,
                           Stride<_16,Stride< _1,_256,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x8x32_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x8x32_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_16>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _2,  _8>>,
                           Stride<_16,Stride< _1,_256,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U16x8x32_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x8x32_LD_N::PREFETCH, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x8x32_LD_N, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,Shape <_32,  _8>>,
                           Stride<_32,Stride< _1,_512>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_32,  _8>>,
                           Stride<_32,Stride< _1,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
  using CopyInternalType = ushort;
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x16x32_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x16x32_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_16>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _2, _16>>,
                           Stride<_16,Stride< _1,_256,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U16x16x32_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x16x32_LD_N::PREFETCH, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x16x32_LD_N, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,Shape <_32, _16>>,
                           Stride<_32,Stride< _1,_512>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_32, _16>>,
                           Stride<_32,Stride< _1,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
  using CopyInternalType = ushort;
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x32x32_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x32x32_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_16>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _2, _32>>,
                           Stride<_16,Stride< _1,_256,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U16x32x32_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x32x32_LD_N::PREFETCH, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x32x32_LD_N, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,Shape <_32, _32>>,
                           Stride<_32,Stride< _1,_512>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_32, _32>>,
                           Stride<_32,Stride< _1,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
  using CopyInternalType = ushort;
};

template <class... args_t>
struct Copy_Traits<XE_2D_TF32x1x8_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_TF32x1x8_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <Shape < _8,_2>,_32>,
                           Stride<Stride<_32,_0>, _1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_TF32x1x8_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_TF32x2x8_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_TF32x2x8_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <Shape < _8,  _2>,_32>,
                           Stride<Stride<_32,_256>, _1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_TF32x2x8_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_TF32x4x8_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_TF32x4x8_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <Shape < _8,  _2>,Shape <_32,  _2>>,
                           Stride<Stride<_32,_256>,Stride< _1,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_TF32x4x8_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_TF32x8x8_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_TF32x8x8_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <Shape < _8,  _2>,Shape <_32,  _4>>,
                           Stride<Stride<_32,_256>,Stride< _1,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_TF32x8x8_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_TF32x16x8_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_TF32x16x8_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <Shape < _8,  _2>,Shape <_32,   _8>>,
                           Stride<Stride<_32,_256>,Stride< _1, _512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_TF32x16x8_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_TF32x32x8_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_TF32x32x8_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <Shape < _8,  _2>,Shape <_32, _16>>,
                           Stride<Stride<_32,_256>,Stride< _1,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_TF32x32x8_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_TF32x1x16_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_TF32x1x16_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <Shape < _8,_2>,Shape <_32,  _2>>,
                           Stride<Stride<_32,_0>,Stride< _1,_256>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_TF32x1x16_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_TF32x2x16_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_TF32x2x16_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <Shape < _8,  _2>,Shape <_32,  _2>>,
                           Stride<Stride<_32,_512>,Stride< _1,_256>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_TF32x2x16_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_TF32x4x16_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_TF32x4x16_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <Shape < _8,  _2>,Shape <_32,  _2,   _2>>,
                           Stride<Stride<_32,_512>,Stride< _1,_256,_1024>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_TF32x4x16_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_TF32x8x16_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_TF32x8x16_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <Shape < _8,  _2>,Shape <_32,  _2,   _4>>,
                           Stride<Stride<_32,_512>,Stride< _1,_256,_1024>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_TF32x8x16_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_TF32x16x16_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_TF32x16x16_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <Shape < _8,  _2>,Shape <_32,  _2,  _16>>,
                           Stride<Stride<_32,_512>,Stride< _1,_256,_1024>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_TF32x16x16_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_TF32x32x16_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_TF32x32x16_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <Shape < _8,  _2>,Shape <_32,  _2,  _32>>,
                           Stride<Stride<_32,_512>,Stride< _1,_256,_1024>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_TF32x32x16_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U32x1x16_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U32x1x16_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,_32>,
                           Stride<_32, _1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U32x1x16_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U32x2x16_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U32x2x16_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_32,  _2>>,
                           Stride<_32,Stride< _1,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U32x2x16_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U32x4x16_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U32x4x16_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_32,  _4>>,
                           Stride<_32,Stride< _1,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U32x4x16_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U32x8x16_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U32x8x16_LD_N, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_32,  _8>>,
                           Stride<_32,Stride< _1,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U32x8x16_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U32x16x16_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U32x16x16_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_32, _16>>,
                           Stride<_32,Stride< _1,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U32x16x16_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U32x32x16_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U32x32x16_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_32, _32>>,
                           Stride<_32,Stride< _1,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U32x32x16_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x32x16_LD_V, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U8x32x16_LD_V, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_8>,
                           Stride< _0,_1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_8,  _4,  _8>>,
                           Stride< _8,Stride<_1,_128,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U8x32x16_LD_V, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x32x16_LD_V::PREFETCH, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U8x32x16_LD_V, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,Shape <_8, _32>>,
                           Stride< _8,Stride<_1,_128>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_8, _32>>,
                           Stride< _8,Stride<_1,_128>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x32x32_LD_V, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U8x32x32_LD_V, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_8>,
                           Stride< _0,_1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_8,  _4,  _2,  _8>>,
                           Stride< _8,Stride<_1,_256,_128,_1024>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U8x32x32_LD_V, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x32x64_LD_V, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U8x32x64_LD_V, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_8>,
                           Stride< _0,_1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_8,  _4,  _4,   _8>>,
                           Stride< _8,Stride<_1,_512,_128,_2048>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U8x32x64_LD_V, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x16x16_LD_V, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x16x16_LD_V, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_16>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _2,  _8>>,
                           Stride<_16,Stride< _1,_256,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U16x16x16_LD_V, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x32x16_LD_V, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x32x16_LD_V, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_16>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _2, _16>>,
                           Stride<_16,Stride< _1,_256,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U16x32x16_LD_V, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x32x32_LD_V, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x32x32_LD_V, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_64>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _2,  _2,  _16>>,
                           Stride<_16,Stride< _1,_512,_256,_1024>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U16x32x32_LD_V, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x16x32_LD_V, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x16x32_LD_V, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_64>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _2,  _2,   _8>>,
                           Stride<_16,Stride< _1,_512,_256,_1024>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U16x16x32_LD_V, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x16x8_LD_T, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x16x8_LD_T, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_16>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape < _16,Shape <_16, _8>>,
                           Stride<_128,Stride< _1,_16>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_LD_Unpack<XE_2D_U16x16x8_LD_T, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x16x16_LD_T, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x16x16_LD_T, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_16>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape < _16,Shape <_16,_16>>,
                           Stride<_256,Stride< _1,_16>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_LD_Unpack<XE_2D_U16x16x16_LD_T, args_t...>(args...) {}
};

// template<class... args_t>
// struct Copy_Traits<XE_2D_U32x16x1_LD_T, args_t...>
//     : XE_2D_LD_Unpack<XE_2D_U32x16x1_LD_T, args_t...> {
//   // Logical thread id to thread idx
//   using ThrID = Layout<_16>;
//   // Map from (src-thr,src-val) to bit
//   using SrcLayout = Layout<Shape <_16,_32>,
//                            Stride< _0, _1>>;
//   // Map from (dst-thr,dst-val) to bit
//   using DstLayout = Layout<Shape <_16,_32>,
//                            Stride<_32, _1>>;
//   // Reference map from (thr,val) to bit
//   using RefLayout = DstLayout;
// };

template <class... args_t>
struct Copy_Traits<XE_2D_U32x16x2_LD_T, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U32x16x2_LD_T, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_32, _2>>,
                           Stride<_64,Stride< _1,_32>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U32x16x2_LD_T, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U32x16x4_LD_T, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U32x16x4_LD_T, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_32, _4>>,
                           Stride<_64,Stride< _1,_32>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U32x16x4_LD_T, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U32x16x8_LD_T, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U32x16x8_LD_T, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape < _16,Shape <_32, _8>>,
                           Stride<_128,Stride< _1,_32>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U32x16x8_LD_T, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U32x16x8_LD_T::PREFETCH, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U32x16x8_LD_T, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <Shape < _8,_2>,Shape <_32, _16>>,
                           Stride<Stride<_32,_0>,Stride< _1,_256>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <Shape < _8,_2>,Shape <_32, _16>>,
                           Stride<Stride<_32,_0>,Stride< _1,_256>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
};

template <class... args_t>
struct Copy_Traits<XE_2D_U64x8x1_LD_T, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U64x8x1_LD_T, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_64>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <Shape < _8,_2>,_64>,
                           Stride<Stride<_64,_0>, _1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U64x8x1_LD_T, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U64x8x2_LD_T, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U64x8x2_LD_T, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_64>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <Shape <  _8,_2>,Shape <_64, _2>>,
                           Stride<Stride<_128,_0>,Stride< _1,_64>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U64x8x2_LD_T, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U64x8x4_LD_T, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U64x8x4_LD_T, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_64>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <Shape <  _8,_2>,Shape <_64, _4>>,
                           Stride<Stride<_256,_0>,Stride< _1,_64>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U64x8x4_LD_T, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x2x32_ST_N, args_t...>
    : XE_2D_ST_Unpack<XE_2D_U8x2x32_ST_N, args_t...> {
  using Shape_MN = Shape<_2,_32>;
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,Shape <_16,  _2>>,
                           Stride<_16,Stride< _1,_256>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,_8>,
                           Stride< _0,_1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_ST_Unpack<XE_2D_U8x2x32_ST_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x1x16_ST_N, args_t...>
    : XE_2D_ST_Unpack<XE_2D_U8x1x16_ST_N, args_t...> {
  using Shape_MN = Shape<_1, _16>;
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_8>,
                           Stride< _8,_1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,_8>,
                           Stride< _0,_1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_ST_Unpack<XE_2D_U8x1x16_ST_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x2x16_ST_N, args_t...>
    : XE_2D_ST_Unpack<XE_2D_U8x2x16_ST_N, args_t...> {
  using Shape_MN = Shape<_2, _16>;
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,Shape <_8,  _2>>,
                           Stride< _8,Stride<_1,_128>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,_8>,
                           Stride< _0,_1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_ST_Unpack<XE_2D_U8x2x16_ST_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x4x16_ST_N, args_t...>
    : XE_2D_ST_Unpack<XE_2D_U8x4x16_ST_N, args_t...> {
  using Shape_MN = Shape<_4, _16>;
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,Shape <_8,  _4>>,
                           Stride< _8,Stride<_1,_128>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,_8>,
                           Stride< _0,_1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x8x16_ST_N, args_t...>
    : XE_2D_ST_Unpack<XE_2D_U8x8x16_ST_N, args_t...> {
  using Shape_MN = Shape<_8, _16>;
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,Shape <_8,  _8>>,
                           Stride< _8,Stride<_1,_128>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,_8>,
                           Stride< _0,_1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;
  using CopyInternalType = uint8_t;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_ST_Unpack<XE_2D_U8x8x16_ST_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x8x32_ST_N, args_t...>
    : XE_2D_ST_Unpack<XE_2D_U8x8x32_ST_N, args_t...> {
  using Shape_MN = Shape<_8, _32>;
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout =
      Layout<Shape<_16, Shape<_8, _16>>, Stride<_0, Stride<_0, _1>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,_8>,
                           Stride< _0,_1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;
  using CopyInternalType = uint8_t;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_ST_Unpack<XE_2D_U8x8x32_ST_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x1x16_ST_N, args_t...>
    : XE_2D_ST_Unpack<XE_2D_U16x1x16_ST_N, args_t...> {
  using Shape_MN = Shape<_1, _16>;
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_16>,
                           Stride<_16, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,_16>,
                           Stride< _0, _1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_ST_Unpack<XE_2D_U16x1x16_ST_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x2x16_ST_N, args_t...>
    : XE_2D_ST_Unpack<XE_2D_U16x2x16_ST_N, args_t...> {
  using Shape_MN = Shape<_2, _16>;
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,Shape <_16,  _2>>,
                           Stride<_16,Stride< _1,_256>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,_16>,
                           Stride< _0, _1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_ST_Unpack<XE_2D_U16x2x16_ST_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x4x16_ST_N, args_t...>
    : XE_2D_ST_Unpack<XE_2D_U16x4x16_ST_N, args_t...> {
  using Shape_MN = Shape<_4, _16>;
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,Shape <_16,  _4>>,
                           Stride<_16,Stride< _1,_256>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,_16>,
                           Stride< _0, _1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_ST_Unpack<XE_2D_U16x4x16_ST_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x8x16_ST_N, args_t...>
    : XE_2D_ST_Unpack<XE_2D_U16x8x16_ST_N, args_t...> {
  using Shape_MN = Shape<_8, _16>;
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,Shape <_16,  _8>>,
                           Stride<_16,Stride< _1,_256>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,_16>,
                           Stride< _0, _1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_ST_Unpack<XE_2D_U16x8x16_ST_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U32x1x16_ST_N, args_t...>
    : XE_2D_ST_Unpack<XE_2D_U32x1x16_ST_N, args_t...> {
  using Shape_MN = Shape<_1, _16>;
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride<_32, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_ST_Unpack<XE_2D_U32x1x16_ST_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U32x2x16_ST_N, args_t...>
    : XE_2D_ST_Unpack<XE_2D_U32x2x16_ST_N, args_t...> {
  using Shape_MN = Shape<_2, _16>;
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,Shape <_32,  _2>>,
                           Stride<_32,Stride< _1,_512>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_ST_Unpack<XE_2D_U32x2x16_ST_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U32x4x16_ST_N, args_t...>
    : XE_2D_ST_Unpack<XE_2D_U32x4x16_ST_N, args_t...> {
  using Shape_MN = Shape<_4, _16>;
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,Shape <_32,  _4>>,
                           Stride<_32,Stride< _1,_512>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_ST_Unpack<XE_2D_U32x4x16_ST_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U32x8x16_ST_N, args_t...>
    : XE_2D_ST_Unpack<XE_2D_U32x8x16_ST_N, args_t...> {
  using Shape_MN = Shape<_8, _16>;
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,Shape <_32,  _8>>,
                           Stride<_32,Stride< _1,_512>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_ST_Unpack<XE_2D_U32x8x16_ST_N, args_t...>(args...) {}
};

template <class Copy, class GEngine, class GLayout>
auto make_xe_2d_copy(Tensor<GEngine, GLayout> gtensor) {
  using GTensor = Tensor<GEngine, GLayout>;
  using Traits = Copy_Traits<Copy, GTensor>;
  // Traits traits {gtensor};
  return Copy_Atom<Traits, typename GEngine::value_type>{gtensor};
}

template<class S, class D>
struct Copy_Traits<XE_ATOMIC<S, D>> {
  // Logical thread id to thread idx (one-thread)
  using ThrID = Layout<_1>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,Int<sizeof_bits<S>::value>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,Int<sizeof_bits<D>::value>>>;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;
};

template<class S, class D>
struct Copy_Traits<XE_1D_LDSM<S, D>> {
    // Logical thread id to thread idx
    using ThrID = Layout<_16>;
    // Map from (src-thr,src-val) to bit
    using SrcLayout = Layout<Shape<_16, Int<sizeof_bits<S>::value>>, Stride<_0, _1>>;
    // Map from (dst-thr,dst-val) to bit
    using DstLayout = Layout<Shape<_16, Shape<Int<sizeof_bits<D>::value / sizeof_bits<S>::value>, Int<sizeof_bits<S>::value>>>,
                             Stride<Int<sizeof_bits<S>::value>, Stride<Int<sizeof_bits<S>::value * 16>, _1>>>;
    // Reference map from (thr,val) to bit
    using RefLayout = DstLayout;
};

template<class S, class D>
struct Copy_Traits<XE_1D_LOAD_GLOBAL<S, D>> {
    // Logical thread id to thread idx
    using ThrID = Layout<_16>;
    // Map from (src-thr,src-val) to bit
    using SrcLayout = Layout<Shape<_16, Int<sizeof_bits<S>::value>>, Stride<_0, _1>>;
    // Map from (dst-thr,dst-val) to bit
    using DstLayout = Layout<Shape <_16, Shape<Int<sizeof_bits<D>::value / sizeof_bits<S>::value>, Int<sizeof_bits<S>::value>>>,
                             Stride<Int<sizeof_bits<S>::value>, Stride<Int<sizeof_bits<S>::value * 16>, _1>>>;
    // Reference map from (thr,val) to bit
    using RefLayout = DstLayout;
};

template<class S, class D>
struct Copy_Traits<PREFETCH<S, D>> {
    // Logical thread id to thread idx
    using ThrID = Layout<_16>;
    // Map from (src-thr,src-val) to bit
    using SrcLayout = Layout<Shape<_16, Int<sizeof_bits<S>::value>>, Stride<_0, _1>>;
    // Map from (dst-thr,dst-val) to bit
    using DstLayout = Layout<Shape <                       _16,Int<sizeof_bits<D>::value>>,
                             Stride<Int<sizeof_bits<D>::value>,                        _1>>;
    // Reference map from (thr,val) to bit
    using RefLayout = DstLayout;

    template <class... CopyArgs>
    CUTE_HOST_DEVICE
    Copy_Traits(Copy_Traits<CopyArgs...> const& traits) {}
};

template<class S, class D>
struct Copy_Traits<XE_1D_STSM<S, D>> {
    // Logical thread id to thread idx
    using ThrID = Layout<_16>;
    // Map from (src-thr,src-val) to bit
    using SrcLayout = Layout<Shape<_16, Shape<Int<sizeof_bits<S>::value / sizeof_bits<D>::value>, Int<sizeof_bits<D>::value>>>,
                             Stride<Int<sizeof_bits<D>::value>, Stride<Int<sizeof_bits<D>::value * 16>, _1>>>;
    // Map from (dst-thr,dst-val) to bit
    using DstLayout = Layout<Shape<_16, Int<sizeof_bits<D>::value>>, Stride<_0, _1>>;
    // Reference map from (thr,val) to bit
    using RefLayout = SrcLayout;
};

template<class S, class D>
struct Copy_Traits<XE_1D_STORE_GLOBAL<S, D>> {
    // Logical thread id to thread idx
    using ThrID = Layout<_16>;
    // Map from (src-thr,src-val) to bit
    using SrcLayout = Layout<Shape<_16, Shape<Int<sizeof_bits<S>::value / sizeof_bits<D>::value>, Int<sizeof_bits<D>::value>>>,
                             Stride<Int<sizeof_bits<D>::value>, Stride<Int<sizeof_bits<D>::value * 16>, _1>>>;
    // Map from (dst-thr,dst-val) to bit
    using DstLayout = Layout<Shape<_16, Int<sizeof_bits<D>::value>>, Stride<_0, _1>>;
    // Reference map from (thr,val) to bit
    using RefLayout = SrcLayout;
};

namespace detail
{
  template<class PrefetchTileSize, class dtype>
  auto prefetch_selector(void* ptr = nullptr, int32_t width = 0, int32_t height = 0, int32_t pitch = 0) {
    if constexpr (get<0>(PrefetchTileSize{}) == 1) {
      using prefetch_trait = Copy_Traits<XE_2D_U8x1x64_LD_N>;
      using prefetch_atom = Copy_Atom<prefetch_trait, dtype>;
      return make_tiled_copy(prefetch_atom{}.with(static_cast<dtype const*>(ptr), width, height, pitch),
                                           Layout<Shape<_1, _16>>{},
                                           Layout<Shape<_2, _2>>{});
    }
    else if constexpr (get<0>(PrefetchTileSize{}) == 2) {
      using prefetch_trait = Copy_Traits<XE_2D_U8x2x64_LD_N>;
      using prefetch_atom = Copy_Atom<prefetch_trait, dtype>;
      return make_tiled_copy(prefetch_atom{}.with(static_cast<dtype const*>(ptr), width, height, pitch),
                                           Layout<Shape<_1, _16>>{},
                                           Layout<Shape<_2, _2, _2>>{});
    }
    else if constexpr (get<0>(PrefetchTileSize{}) == 4) {
      using prefetch_trait = Copy_Traits<XE_2D_U8x4x64_LD_N>;
      using prefetch_atom = Copy_Atom<prefetch_trait, dtype>;
      return make_tiled_copy(prefetch_atom{}.with(static_cast<dtype const*>(ptr), width, height, pitch),
                                           Layout<Shape<_1, _16>>{},
                                           Layout<Shape<_2, _2, _4>>{});
    }
    else if constexpr (get<0>(PrefetchTileSize{}) == 8) {
      using prefetch_trait = Copy_Traits<XE_2D_U8x8x64_LD_N>;
      using prefetch_atom = Copy_Atom<prefetch_trait, dtype>;
      return make_tiled_copy(prefetch_atom{}.with(static_cast<dtype const*>(ptr), width, height, pitch),
                                           Layout<Shape<_1, _16>>{},
                                           Layout<Shape<_2, _2, _8>>{});
    }
    else if constexpr (get<0>(PrefetchTileSize{}) == 16) {
      // static_assert(false);
      using prefetch_trait = Copy_Traits<XE_2D_U8x16x64_LD_N>;
      using prefetch_atom = Copy_Atom<prefetch_trait, dtype>;
      return make_tiled_copy(prefetch_atom{}.with(static_cast<dtype const*>(ptr), width, height, pitch),
                                           Layout<Shape<_1, _16>>{},
                                           Layout<Shape<_2, _2, _16>>{});
    }
    else if constexpr (get<0>(PrefetchTileSize{}) == 32) {
      using prefetch_trait = Copy_Traits<XE_2D_U8x32x64_LD_N>;
      using prefetch_atom = Copy_Atom<prefetch_trait, dtype>;
      return make_tiled_copy(prefetch_atom{}.with(static_cast<dtype const*>(ptr), width, height, pitch),
                                           Layout<Shape<_1, _16>>{},
                                           Layout<Shape<_2, _2, _32>>{});
    }
    else{
      static_assert(dependent_false<PrefetchTileSize> && "Invalid PrefetchTileSize[0]");
    }
  }
} // end namespace detail

template <class TiledCopy, class ThrIdx>
struct Xe2DThrCopy : ThrCopy<TiledCopy, ThrIdx> {

  CUTE_HOST_DEVICE
  Xe2DThrCopy(ThrIdx const& thr_idx) : ThrCopy<TiledCopy, ThrIdx> (thr_idx) {}

  template <class DTensor>
  CUTE_HOST_DEVICE
  auto
  retile_D(DTensor&& dtensor) {
    if constexpr (TiledCopy::is_nkl) {
      return retile_B(dtensor);
    } else {
      return retile_A(dtensor);
    }
  }

  template <class DTensor>
  CUTE_HOST_DEVICE static
  auto
  retile_A(DTensor&& dtensor) {
    auto tmp = ThrCopy<TiledCopy, ThrIdx>::retile_D(dtensor);
    return make_tensor(static_cast<decltype(tmp) &&>(tmp).data(),
                           tmp.shape());
  }

  template <class DTensor>
  CUTE_HOST_DEVICE static
  auto
  retile_B(DTensor&& dtensor) {
    auto b_tensor = make_tensor(dtensor.data(),
                               make_shape(size<0>(dtensor.shape()),
                                         size<2>(dtensor.shape()),
                                         size<1>(dtensor.shape())));
    auto tmp = ThrCopy<TiledCopy, ThrIdx>::retile_D(b_tensor);
    return make_tensor(static_cast<decltype(tmp) &&>(tmp).data(),
                       make_shape(size<0>(tmp.shape()),
                                  size<2>(tmp.shape()),
                                  size<1>(tmp.shape())));
  }
};

template <class Copy_Atom,
          class LayoutCopy_TV,  // (tid,vid) -> coord   [Need not be 2D...]
          class ShapeTiler_MN>  // coord space
struct Xe2DTiledCopy : TiledCopy<Copy_Atom, LayoutCopy_TV, ShapeTiler_MN>{

  template <class ThrIdx,
            __CUTE_REQUIRES(is_integral<ThrIdx>::value)>
  CUTE_HOST_DEVICE
  auto
  get_slice(ThrIdx const& thr_idx)
  {
    return Xe2DThrCopy<Xe2DTiledCopy, ThrIdx>(thr_idx);
  }
};

template <class... Args,
          class ThrLayout,
          class ValLayout = Layout<_1>>
CUTE_HOST_DEVICE
auto
make_xe_2d_copy(Copy_Atom<Args...> const& copy_atom,
                ThrLayout          const& thr_layout = {},     // (m,n) -> thr_idx
                ValLayout          const& val_layout = {})     // (m,n) -> val_idx
{
  // Take the raked_products to compute the Layout_MN
  // (M,N) -> (thr_idx, val_idx)
  auto layout_mn = raked_product(thr_layout, val_layout);
  // (thr_idx, val_idx) -> (M,N)
  auto layout_tv = right_inverse(layout_mn).with_shape(make_shape(size(thr_layout), size(val_layout)));
  // Tiler for extracting relevant elements
  // (M,N) -> tensor coord
  auto tiler = product_each(shape(layout_mn));

#if 0
  print("thr_layout: "); print(thr_layout); print("\n");
  print("val_layout: "); print(val_layout); print("\n");
  print("layout_mn : "); print(layout_mn);  print("\n");
  print("layout_tv : "); print(layout_tv);  print("\n");
  print("tiler     : "); print(tiler);      print("\n");
#endif

  return Xe2DTiledCopy<Copy_Atom<Args...>, decltype(layout_tv), decltype(tiler)>{copy_atom};
}

} // end namespace cute
