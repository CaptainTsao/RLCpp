//
// Created by transwarp on 2021/7/15.
//

#ifndef EXAMPLE_APP_INCLUDE_ALGORITHM_PROXIMAL_POLICY_OPTIMIZATION_HPP_
#define EXAMPLE_APP_INCLUDE_ALGORITHM_PROXIMAL_POLICY_OPTIMIZATION_HPP_

#include <torch/torch.h>

#include "algorithm.hpp"

namespace RLCpp {
class Policy;
class RolloutStorage;

class ProximalPolicyOptimization : public Algorithm {
 private:
  Policy &policy_;
  float actor_loss_coef_, value_loss_coef_, entropy_coef_,
      max_grad_norm_, original_learning_rate_,
      original_clip_param_, kl_target_;
  int num_epoch_, num_mini_batch_;
  std::unique_ptr<torch::optim::Adam> optimizer_;

 public:
  ProximalPolicyOptimization(Policy &policy,
                             float clip_param,
                             int num_epoch,
                             int num_mini_batch,
                             float actor_loss_coef,
                             float value_loss_coef,
                             float entropy_coef,
                             float learning_rate,
                             float epsilon = 1e-8,
                             float max_grad_norm = 0.5,
                             float kl_target = 0.01);

  std::vector<UpdateDatum> Update(RolloutStorage &rollout_storage, float decay_level = 1);
};

} // namespace RLCpp



#endif //EXAMPLE_APP_INCLUDE_ALGORITHM_PROXIMAL_POLICY_OPTIMIZATION_HPP_
