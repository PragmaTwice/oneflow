#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_REDUCE_IMPL_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_REDUCE_IMPL_H_

#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/switch_func.h"
#include "oneflow/core/ndarray/ndarray_reduce_core.h"
#include "oneflow/core/ndarray/xpu_ndarray_assign.h"

namespace oneflow {

#define DECLARE_NDARRAY_REDUCE_IMPL(struct_name)                                                   \
  template<DeviceType device_type, typename T, const T (*binary_func)(const T, const T)>           \
  struct struct_name final {                                                                       \
    static bool Matched(const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x);               \
    static void Reduce(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x, \
                       const XpuVarNdarray<T>& tmp_storage);                                       \
  }
DECLARE_NDARRAY_REDUCE_IMPL(NdarrayScalarReduce);
DECLARE_NDARRAY_REDUCE_IMPL(NdarrayMatrixRowReduce);
DECLARE_NDARRAY_REDUCE_IMPL(NdarrayMatrixColReduce);
#undef DECLARE_NDARRAY_REDUCE_IMPL

template<DeviceType device_type, typename T, const T (*binary_func)(const T, const T)>
struct NdarrayNoReduce final {
  static bool Matched(const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x) {
    return x.shape() == y.shape();
  }
  static void Reduce(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x,
                     const XpuVarNdarray<T>& tmp_storage) {
    XpuNdArrayAssign<device_type, T>::Assign(ctx, y, x);
  }
};

template<DeviceType device_type, typename T, const T (*binary_func)(const T, const T)>
struct NdarrayDefaultReduce final {
  static void Reduce(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x,
                     const XpuVarNdarray<T>& tmp_storage) {
    return SwitchReduce(SwitchCase(y.shape().NumAxes()), ctx, y, x, tmp_storage);
  }

 private:
#define DEFINE_NDARRAY_REDUCE(func_name, NDIMS) func_name<NDIMS>
  DEFINE_STATIC_SWITCH_FUNC(void, Reduce, DEFINE_NDARRAY_REDUCE, MAKE_NDIM_CTRV_SEQ(DIM_SEQ));
#undef DEFINE_NDARRAY_REDUCE

  template<int NDIMS>
  static void Reduce(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x,
                     const XpuVarNdarray<T>& tmp_storage) {
    XpuVarNdarray<T> storage(x.shape(), tmp_storage.ptr());
    XpuShape cur_shape(x.shape());
    CHECK_EQ(y.shape().NumAxes(), x.shape().NumAxes());
    CHECK(x.shape() != y.shape());
    XpuNdArrayAssign<device_type, T>::Assign(ctx, storage, x);
    for (int i = 0; i < x.shape().NumAxes(); ++i) {
      if (y.shape().At(i) == x.shape().At(i)) { continue; }
      CHECK_EQ(y.shape().At(i), 1);
      CHECK_GT(x.shape().At(i), y.shape().At(i));
      ImplaceReduceAxis<NDIMS>(ctx, i, storage, &cur_shape);
    }
    XpuReducedNdarray<T, NDIMS> reduced(y.shape(), storage);
    XpuNdArrayAssign<device_type, T>::template Assign<NDIMS>(ctx, y, reduced);
  }

  template<int NDIMS>
  static void ImplaceReduceAxis(DeviceCtx* ctx, int axis, const XpuVarNdarray<T>& implace,
                                XpuShape* cur_shape) {
    int64_t target_elem_num = cur_shape->ElemNum() / cur_shape->At(axis);
    while (cur_shape->At(axis) > 1) {
      int64_t shrink = 8 + std::sqrt(target_elem_num);
      XpuReducedNdarray<T, NDIMS> from(*cur_shape, implace);
      int64_t new_dim_value = (cur_shape->At(axis) + (shrink - 1)) / shrink;
      cur_shape->Set(axis, new_dim_value);
      XpuReducedNdarray<T, NDIMS> to(*cur_shape, implace);
      NdArrayReduceCoreWrapper<device_type, T, NDIMS, binary_func>::ReduceAxis(ctx, to, from, axis);
    }
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_REDUCE_IMPL_H_
