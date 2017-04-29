#include "operator/concat_op.h"
#include "operator/operator_manager.h"

namespace oneflow {

void ConcatOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_concat_conf());
  mut_op_conf() = op_conf;

  for (int i = 0; i < op_conf.concat_conf().in_size(); ++i) {
    std::string ibn = "in_" + std::to_string(i);
    EnrollInputBn(ibn);
    CHECK(ibn2lbn_.emplace(ibn, op_conf.concat_conf().in(i)).second);
  }
  EnrollOutputBn("out");
}
std::string ConcatOp::GetValueFromPbOpConf(const std::string& k) const {
  return GetValueFromPbMessage(op_conf().concat_conf(), k);
}

void ConcatOp::InferShape4ObAndDtbFromIb() const {
  std::vector<int64_t> vec;
  int axis = op_conf().concat_conf().axis();
  for (int i = 0; i < input_bns().size(); ++i) {
    Shape* in_shape_tmp = GetShapePtr(input_bns()[i]);
    if (i == 0) {
      vec = in_shape_tmp->dim_vec();
    } else {
      for (int j = 0; j < in_shape_tmp->NumAxes(); ++j) {
        if (j == axis) {
          vec[j] += in_shape_tmp->At(j);
        } else {
          CHECK_EQ(vec[j], in_shape_tmp->At(j));
        }
      }
    }
  }
  *GetShapePtr(SoleObn()) = Shape(vec);
}
REGISTER_OP(OperatorConf::kConcatConf, ConcatOp);

} // namespace oneflow
