#ifndef ONEFLOW_MLIRONEFLOWTRANSLATION_H
#define ONEFLOW_MLIRONEFLOWTRANSLATION_H

#include "oneflow/core/job/job.pb.h"
#include <functional>
namespace mlir {

class RoundTripOneFlowJobWrapperInterface {
 public:
  virtual const ::oneflow::Job* job() const = 0;
  virtual const ::oneflow::ParallelConf& ParallelConf4OpName(const std::string& op_name) const = 0;
  virtual std::pair<std::vector<std::string>, std::vector<std::string>> InputBns4OpName(
      const std::string& op_name) const = 0;
  virtual std::vector<std::string> OutputLbns4OpName(const std::string& op_name) const = 0;
};

void RoundTripOneFlowJob(
    RoundTripOneFlowJobWrapperInterface& job_wrapper,
    std::function<bool(::oneflow::Job* job, std::string& reason)> is_legit_job);
void registerFromOneFlowJobTranslation();

}  // namespace mlir

#endif /* ONEFLOW_MLIRONEFLOWTRANSLATION_H */
