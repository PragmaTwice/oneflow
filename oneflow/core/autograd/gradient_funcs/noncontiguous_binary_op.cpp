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
#include <glog/logging.h>
#include "oneflow/core/common/just.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/functional_api.yaml.h"

namespace oneflow {
namespace one {

struct NonContiguousBinaryOpCaptureState : public AutoGradCaptureState {
  bool lhs_requires_grad = false;
  bool rhs_requires_grad = false;
  std::string op = "add";
  bool inplace = false;
};

class NonContiguousBinaryOp : public OpExprGradFunction<NonContiguousBinaryOpCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(NonContiguousBinaryOpCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const NonContiguousBinaryOpCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> NonContiguousBinaryOp::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> NonContiguousBinaryOp::Capture(NonContiguousBinaryOpCaptureState* ctx,
                                           const TensorTuple& inputs, const TensorTuple& outputs,
                                           const AttrMap& attrs) const {
  ctx->lhs_requires_grad = inputs.at(0)->requires_grad();
  ctx->rhs_requires_grad = inputs.at(1)->requires_grad();
  if (!ctx->lhs_requires_grad && !ctx->rhs_requires_grad) { return Maybe<void>::Ok(); }

  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->inplace = JUST(composed_attrs.GetAttr<bool>("inplace"));
  ctx->op = JUST(composed_attrs.GetAttr<std::string>("op"));
  if (ctx->inplace && ctx->rhs_requires_grad) {
    CHECK_OR_RETURN(ctx->op == "add" || ctx->op == "sub")
        << "when inplace and rhs requires grad, op should be add/sub";
  }
  ctx->SaveTensorForBackward(inputs.at(0));
  ctx->SaveTensorForBackward(inputs.at(1));
  return Maybe<void>::Ok();
}

Maybe<void> NonContiguousBinaryOp::Apply(const NonContiguousBinaryOpCaptureState* ctx,
                                         const TensorTuple& out_grads,
                                         TensorTuple* in_grads) const {
  if (!ctx->lhs_requires_grad && !ctx->rhs_requires_grad) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
  in_grads->resize(2);
  auto lhs = ctx->SavedTensors().at(0);
  auto rhs = ctx->SavedTensors().at(1);
  auto ret = JUST(functional::NonContiguousBinaryOpGrad(out_grads.at(0), lhs, rhs, ctx->op, false));
  if (ctx->lhs_requires_grad) in_grads->at(0) = ret->at(0);
  if (ctx->rhs_requires_grad) in_grads->at(1) = ret->at(1);
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("noncontiguous_binary_op", NonContiguousBinaryOp);

}  // namespace one
}  // namespace oneflow
