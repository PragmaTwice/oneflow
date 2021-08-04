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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/device.h"

namespace oneflow {

Maybe<Symbol<Device>> DeviceInferFn(user_op::DeviceInferContext* ctx) {
  const auto& input_device = ctx->InputTensorDevice4ArgNameAndIndex("in", 0);
  *ctx->OutputTensorDevice4ArgNameAndIndex("out", 0) = input_device;
  if (input_device->type() == "cuda" || input_device->type() == "gpu") {
    static thread_local const auto& nccl_device = Device::New("nccl");
    return nccl_device;
  } else if (input_device->type() == "cpu") {
    return input_device;
  } else {
    UNIMPLEMENTED_THEN_RETURN();
  }
}

REGISTER_NO_GRAD_USER_OP("eager_nccl_all_reduce")
    .Input("in")
    .Output("out")
    .Attr<std::string>("parallel_conf")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
      return Maybe<void>::Ok();
    })
    .SetDeviceInferFn(DeviceInferFn)
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      UNIMPLEMENTED_THEN_RETURN() << "consistent tensor are not supported";
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("eager_nccl_broadcast")
    .Input("in")
    .Output("out")
    .Attr<std::string>("parallel_conf")
    .Attr<int64_t>("root", 0)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
      return Maybe<void>::Ok();
    })
    .SetDeviceInferFn(DeviceInferFn)
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      UNIMPLEMENTED_THEN_RETURN() << "consistent tensor are not supported";
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("eager_nccl_reduce")
    .Input("in")
    .Output("out")
    .Attr<std::string>("parallel_conf")
    .Attr<int64_t>("root", 0)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
      return Maybe<void>::Ok();
    })
    .SetDeviceInferFn(DeviceInferFn)
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      UNIMPLEMENTED_THEN_RETURN() << "consistent tensor are not supported";
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
      return Maybe<void>::Ok();
    });
}  // namespace oneflow
