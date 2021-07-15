//
// Created by transwarp on 2021/7/15.
//

#include "../../include/algorithm/algorithm.hpp"
#include "../../include/algorithm/proximal_policy_optimization.hpp"
#include "../../include/model/mlp_base.hpp"
#include "../../include/model/policy.hpp"

namespace RLCpp {

ProximalPolicyOptimization::ProximalPolicyOptimization(Policy &policy,
                                                       float clip_param,
                                                       int num_epoch,
                                                       int num_mini_batch,
                                                       float actor_loss_coef,
                                                       float value_loss_coef,
                                                       float entropy_coef,
                                                       float learning_rate,
                                                       float epsilon,
                                                       float max_grad_norm,
                                                       float kl_target) :
    policy_(policy),
    actor_loss_coef_(actor_loss_coef),
    value_loss_coef_(value_loss_coef),
    entropy_coef_(entropy_coef),
    max_grad_norm_(max_grad_norm),
    original_learning_rate_(learning_rate),
    original_clip_param_(clip_param),
    kl_target_(kl_target),
    optimizer_(std::make_unique<torch::optim::Adam>(
        policy->parameters(),
        torch::optim::AdamOptions(learning_rate).eps(epsilon)
        ))
    {

}
std::vector<UpdateDatum> ProximalPolicyOptimization::Update(RolloutStorage &rollout_storage, float decay_level) {
  return std::vector<UpdateDatum>();
}
} // namespace RLCpp