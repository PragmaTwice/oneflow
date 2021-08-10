/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_API_PYTHON_FUNCTIONAL_UNPACK_CALL_H_
#define ONEFLOW_API_PYTHON_FUNCTIONAL_UNPACK_CALL_H_

#include "oneflow/api/python/functional/python_arg.h"

#include <tuple>
#include "oneflow/api/python/framework/throw.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/common/function_traits.h"
#include "oneflow/core/common/cplusplus_14.h"

#include "oneflow/core/profiler/profiler.h"

namespace oneflow {
namespace one {
namespace functional {

namespace detail {

template<typename F, typename R>
struct unpack_call_dispatcher {
  template<size_t... I, size_t N>
  static R apply(const F& f, const std::array<PythonArg, N>& args, std::index_sequence<I...>) {
    OF_PROFILER_RANGE_PUSH("functor call");
    const auto& res =
        f(args[I]
              .template As<oneflow::detail::remove_cvref_t<typename std::tuple_element<
                  I, typename function_traits<F>::args_type>::type>>()...);
    OF_PROFILER_RANGE_POP();
    return res;
  }
};

template<typename F, typename R>
struct unpack_call {
  template<size_t N>
  static R apply(const F& f, const std::array<PythonArg, N>& args) {
    constexpr size_t nargs = function_traits<F>::nargs;
    static_assert(nargs == N, "parameter number of `f` should equal to size of `args`");
    return unpack_call_dispatcher<F, R>::apply(f, args, std::make_index_sequence<nargs>{});
  }
};

#define INSTANCE_MAYBE_UNPACK_CALL(K, R, return_fn)                                         \
  template<typename F>                                                                      \
  struct unpack_call<F, K> {                                                                \
    template<size_t N>                                                                      \
    static R apply(const F& f, const std::array<PythonArg, N>& args) {                      \
      constexpr size_t nargs = function_traits<F>::nargs;                                   \
      static_assert(nargs == N, "parameter number of `f` should equal to size of `args`");  \
      return (return_fn)(                                                                   \
          unpack_call_dispatcher<F, K>::apply(f, args, std::make_index_sequence<nargs>{})); \
    }                                                                                       \
  };

INSTANCE_MAYBE_UNPACK_CALL(Maybe<one::Tensor>, std::shared_ptr<one::Tensor>,
                           ([](const Maybe<one::Tensor>& t) { return t.GetPtrOrThrow(); }));
INSTANCE_MAYBE_UNPACK_CALL(Maybe<one::TensorTuple>, std::shared_ptr<one::TensorTuple>,
                           ([](const Maybe<one::TensorTuple>& t) { return t.GetPtrOrThrow(); }));
INSTANCE_MAYBE_UNPACK_CALL(Maybe<void>, bool, ([](const Maybe<void>& t) {
                             t.GetOrThrow();
                             return true;
                           }));

#undef INSTANCE_MAYBE_UNPACK_CALL

}  // namespace detail

}  // namespace functional
}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_API_PYTHON_FUNCTIONAL_UNPACK_CALL_H_
