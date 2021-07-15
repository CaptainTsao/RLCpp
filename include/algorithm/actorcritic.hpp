//
// Created by transwarp on 2021/7/7.
//

#ifndef EXAMPLE_APP_INCLUDE_ALGORITHM_ACTORCRITIC_HPP_
#define EXAMPLE_APP_INCLUDE_ALGORITHM_ACTORCRITIC_HPP_
#include <torch/torch.h>
#include "algorithm.hpp"

namespace RLCpp {
class Policy;
class RolloutStorage;
class ActorCritic : public Algorithm {
 private:
  Policy &policy_;
  float actor_loss_coef_, value_loss_coef_, entropy_coef_,
      max_grad_norm_, original_learning_rate_;
  std::unique_ptr<torch::optim::RMSprop> optimizer_;  /* can choose RMSprop */
 public:
  ActorCritic(Policy &policy,
              float actor_loss_coef,
              float value_loss_coef,
              float entropy_coef,
              float learning_rate,
              float epsilon = 1e-8,
              float alpha = 0.99,
              float max_grad_norm = 0.5);

  std::vector<UpdateDatum> Update(RolloutStorage &rollout_storage, float decay_level);
};
}

#endif //EXAMPLE_APP_INCLUDE_ALGORITHM_ACTORCRITIC_HPP_
