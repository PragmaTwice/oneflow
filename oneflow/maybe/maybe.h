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

#ifndef ONEFLOW_MAYBE_MAYBE_H_
#define ONEFLOW_MAYBE_MAYBE_H_

#include <cstddef>
#include <type_traits>

#include "oneflow/maybe/just.h"
#include "oneflow/maybe/variant.h"
#include "oneflow/maybe/optional.h"
#include "oneflow/maybe/error.h"
#include "oneflow/maybe/config.h"

namespace oneflow {

namespace maybe {

struct InPlaceOkType {
  explicit constexpr InPlaceOkType() = default;
};

constexpr InPlaceOkType Ok{};

struct InPlaceErrorType {
  explicit constexpr InPlaceErrorType() = default;
};

constexpr InPlaceErrorType InPlaceError{};

namespace details {

template<typename T, typename E, typename = void>
struct MaybeStorage : Variant<T, E> {
  using Base = Variant<T, E>;

  MaybeStorage(const T& v) : Base(v) {}        // NOLINT(google-explicit-constructor)
  MaybeStorage(T&& v) : Base(std::move(v)) {}  // NOLINT(google-explicit-constructor)

  template<typename... Args>
  explicit MaybeStorage(InPlaceOkType, Args&&... args)
      : Base(InPlaceType<T>, std::forward<Args>(args)...) {}

  template<typename... Args>
  explicit MaybeStorage(InPlaceErrorType, Args&&... args)
      : Base(InPlaceType<E>, std::forward<Args>(args)...) {}

  MaybeStorage(const E& err) : Base(err) {}        // NOLINT(google-explicit-constructor)
  MaybeStorage(E&& err) : Base(std::move(err)) {}  // NOLINT(google-explicit-constructor)

  decltype(auto) Value() & { return this->template Get<T>(); }
  decltype(auto) Value() const& { return this->template Get<T>(); }
  decltype(auto) Value() && { return std::move(*this).template Get<T>(); }

  decltype(auto) Error() & { return this->template Get<E>(); }
  decltype(auto) Error() const& { return this->template Get<E>(); }
  decltype(auto) Error() && { return std::move(*this).template Get<E>(); }

  bool IsOk() const { return this->template Is<T>(); }
};

template<typename T, typename E>
struct MaybeStorage<T, E, std::enable_if_t<std::is_reference<T>::value>>
    : Variant<std::remove_reference_t<T>*, E> {
  static_assert(std::is_lvalue_reference<T>::value, "rvalue reference is not allowed here");

  using PointedType = std::remove_reference_t<T>;
  using UnderlyingType = PointedType*;
  using Base = Variant<UnderlyingType, E>;

  MaybeStorage(T v) : Base(&v) {}  // NOLINT(google-explicit-constructor)

  MaybeStorage(const E& err) : Base(err) {}        // NOLINT(google-explicit-constructor)
  MaybeStorage(E&& err) : Base(std::move(err)) {}  // NOLINT(google-explicit-constructor)

  template<typename... Args>
  explicit MaybeStorage(InPlaceErrorType, Args&&... args)
      : Base(InPlaceType<E>, std::forward<Args>(args)...) {}

  PointedType& Value() { return *this->template Get<UnderlyingType>(); }

  const PointedType& Value() const { return *this->template Get<UnderlyingType>(); }

  decltype(auto) Error() & { return this->template Get<E>(); }
  decltype(auto) Error() const& { return this->template Get<E>(); }
  decltype(auto) Error() && { return std::move(*this).template Get<E>(); }

  bool IsOk() const { return this->template Is<UnderlyingType>(); }
};

template<typename E>
struct MaybeStorage<void, E> : Optional<E> {
  using Base = Optional<E>;

  MaybeStorage(InPlaceOkType) : Base(NullOpt) {}  // NOLINT(google-explicit-constructor)

  MaybeStorage(const E& err) : Base(err) {}        // NOLINT(google-explicit-constructor)
  MaybeStorage(E&& err) : Base(std::move(err)) {}  // NOLINT(google-explicit-constructor)

  template<typename... Args>
  explicit MaybeStorage(InPlaceErrorType, Args&&... args)
      : Base(InPlace, std::forward<Args>(args)...) {}

  void Value() const {}

  decltype(auto) Error() & { return this->Base::Value(); }
  decltype(auto) Error() const& { return this->Base::Value(); }
  decltype(auto) Error() && { return std::move(*this).Base::Value(); }

  bool IsOk() const { return !this->HasValue(); }
};

struct MaybePrivateScope {
  template<typename T>
  static decltype(auto) Value(T&& m) {
    return std::forward<T>(m).Value();
  }

  template<typename T>
  static decltype(auto) StackedError(T&& m) {
    return std::forward<T>(m).StackedError();
  }
};

}  // namespace details

// A type which can be either a value typed T, or a stacked error typed E
template<typename T, typename E>
struct Maybe : private details::MaybeStorage<T, E> {
  static_assert(!std::is_reference<E>::value, "error type cannot be reference");
  static_assert(!(std::is_const<E>::value || std::is_volatile<E>::value),
                "error type cannot be cv-qualified");

  // E must be a stacked error, which implies StackedErrorTraits<E> must exist
  using ErrorTraits = StackedErrorTraits<E>;
  using StackedErrorType = E;
  using ValueType = T;
  using ErrorType = typename ErrorTraits::ErrorType;

 private:
  using Base = details::MaybeStorage<T, E>;

  friend struct details::MaybePrivateScope;
  friend struct details::JustPrivateScope;

 protected:
  decltype(auto) Value() & { return Base::Value(); }
  decltype(auto) Value() const& { return Base::Value(); }
  decltype(auto) Value() && { return std::move(*this).Base::Value(); }

  decltype(auto) StackedError() & { return Base::Error(); }
  decltype(auto) StackedError() const& { return Base::Error(); }
  decltype(auto) StackedError() && { return std::move(*this).Base::Error(); }

  decltype(auto) Error() & { return ErrorTraits::Error(StackedError()); }
  decltype(auto) Error() const& { return ErrorTraits::Error(StackedError()); }
  decltype(auto) Error() && { return ErrorTraits::Error(std::move(*this).StackedError()); }

 public:
  using Base::Base;

  bool IsOk() const { return Base::IsOk(); }
  bool IsErr() const { return !Base::IsOk(); }
  explicit operator bool() const { return IsOk(); }

  decltype(auto) GetStackedError() & {
    OF_MAYBE_ASSERT(IsErr());
    return StackedError();
  }

  decltype(auto) GetStackedError() const& {
    OF_MAYBE_ASSERT(IsErr());
    return StackedError();
  }

  decltype(auto) GetStackedError() && {
    OF_MAYBE_ASSERT(IsErr());
    return std::move(*this).StackedError();
  }

  decltype(auto) GetError() & {
    OF_MAYBE_ASSERT(IsErr());
    return Error();
  }

  decltype(auto) GetError() const& {
    OF_MAYBE_ASSERT(IsErr());
    return Error();
  }

  decltype(auto) GetError() && {
    OF_MAYBE_ASSERT(IsErr());
    return std::move(*this).Error();
  }
};

}  // namespace maybe

}  // namespace oneflow

#endif  // ONEFLOW_MAYBE_MAYBE_H_
