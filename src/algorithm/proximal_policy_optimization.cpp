//
// Created by transwarp on 2021/7/15.
//

#include "algorithm/proximal_policy_optimization.hpp"
#include "model/mlp_base.hpp"
#include "model/policy.hpp"
#include "storage.hpp"
#include "spaces.hpp"
#include "third_party/doctest.hpp"

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
    num_epoch_(num_epoch),
    num_mini_batch_(num_mini_batch),
    optimizer_(std::make_unique<torch::optim::Adam>(
        policy->parameters(),
        torch::optim::AdamOptions(learning_rate).eps(epsilon)
    )) {}

std::vector<UpdateDatum> ProximalPolicyOptimization::Update(RolloutStorage &rollout_storage, float decay_level) {
  /* Decay lr and clip parameter */
  float clip_param = original_clip_param_ * decay_level;
  for (auto &group : optimizer_->param_groups()) {
    if (group.has_options()) {
      auto &options = dynamic_cast<torch::optim::AdamOptions &>(group.options());
      options.lr(options.lr() * decay_level);
    }
  }
  //  Calculate advantages
  auto returns = rollout_storage.get_returns();
  auto value_preds = rollout_storage.get_value_predictions();
  auto advantages = (returns.narrow(0, 0, returns.size(0) - 1),
      value_preds.narrow(0, 0, value_preds.size(0) - 1));

  /* Normalize advantages */
  advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5);

  float total_value_loss = 0.0;
  float total_action_loss{};
  float total_entropy{};
  float kl_divergence{};
  float kl_early_stopped{-1};
  float clip_fraction{};
  int num_updates{0};

  /* Epoch loop */
  for (int epoch = 0; epoch < num_epoch_; ++epoch) {
    /* Shuffle rollouts */
    std::unique_ptr<Generator> data_generator;
    if (policy_->is_recurrent()) {
      data_generator = rollout_storage.recurrent_generator(advantages, num_mini_batch_);
    } else {
      data_generator = rollout_storage.feed_forward_generator(advantages, num_mini_batch_);
    }

    /* Loop through shuffled rollout */
    while (!data_generator->done()) {
      MiniBatch mini_batch = data_generator->next();
      /*Run evaluation on mini-batch */
      auto evaluate_result = policy_->evaluate_actions(
          mini_batch.observations_,
          mini_batch.hidden_states_,
          mini_batch.masks_,
          mini_batch.actions_
      );

      /* Calculate approximate KL divergence for info and early stopping */
      kl_divergence = (mini_batch.action_log_probs_ - evaluate_result[1]).mean().item().toFloat();
      if (kl_divergence > kl_target_ * 1.5) {
        kl_early_stopped = num_updates;
        goto finish_update;
      }
    }
  }

  finish_update:
  /*Update*/
  if (policy_->using_observation_normalizer()) {
    policy_->update_observation_normalizer(rollout_storage.get_observations());
  }
  total_value_loss /= static_cast<float>(num_updates);
  total_action_loss /= static_cast<float>(num_updates);
  total_value_loss /= static_cast<float>(num_updates);
  clip_fraction /= static_cast<float>(num_updates);
  if (kl_early_stopped > -1) {
    return {{"value loss", total_value_loss},
            {"Action loss", total_action_loss},
            {"Clip fraction", clip_fraction},
            {"Entropy", total_entropy},
            {"KL divergence", kl_divergence},
            {"KL divergence early stop update", kl_early_stopped}};
  }
  return std::vector<UpdateDatum>{};
}
} // namespace RLCpp