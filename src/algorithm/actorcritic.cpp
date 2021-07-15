//
// Created by transwarp on 2021/7/8.
//


#include "../../include/algorithm/actorcritic.hpp"
#include "../../include/algorithm/algorithm.hpp"
#include "../../include/model/mlp_base.hpp"
#include "../../include/model/policy.hpp"
#include "../../include/spaces.hpp"
#include "../../include/storage.hpp"

namespace RLCpp {
ActorCritic::ActorCritic(RLCpp::Policy &policy,
                         float actor_loss_coef,
                         float value_loss_coef,
                         float entroy_coef,
                         float learning_rate,
                         float epsilon,
                         float alpha,
                         float max_grad_norm)
    : policy_(policy),
      actor_loss_coef_(actor_loss_coef),
      value_loss_coef_(value_loss_coef),
      entropy_coef_(entroy_coef),
      max_grad_norm_(max_grad_norm),
      original_learning_rate_(learning_rate),
      optimizer_(std::make_unique<torch::optim::RMSprop>(
          policy->parameters(),
          torch::optim::RMSpropOptions(learning_rate).eps(epsilon).alpha(alpha)
      )) {}
std::vector<UpdateDatum> ActorCritic::Update(RolloutStorage &rollout_storage, float decay_level) {
  // Decay learning rate
  // Prep work
  auto full_obs_shape = rollout_storage.get_observations().sizes();
  std::vector<int64_t> obs_shape(full_obs_shape.begin() + 2,
                                 full_obs_shape.end());
  obs_shape.insert(obs_shape.begin(), -1);
  auto action_shape = rollout_storage.get_actions().size(-1);
  auto rewards_shape = rollout_storage.get_rewards().sizes();
  int num_steps = rewards_shape[0];
  int num_processes = rewards_shape[1];

  // Update observation normalizer
  if (policy_->using_observation_normalizer()) {
    policy_->update_observation_normalizer(rollout_storage.get_observations());
  }

  // Run evaluation on rollouts
  auto evaluate_result = policy_->evaluate_actions(
      rollout_storage.get_observations().slice(0, 0, -1).view(obs_shape),
      rollout_storage.get_hidden_states()[0].view({-1, policy_->get_hidden_size()}),
      rollout_storage.get_masks().slice(0, 0, -1).view({-1, 1}),
      rollout_storage.get_actions().view({-1, action_shape}));
  auto values = evaluate_result[0].view({num_steps, num_processes, 1});
  auto action_log_probs = evaluate_result[1].view(
      {num_steps, num_processes, 1});

  // Calculate advantages
  // Advantages aren't normalized (they are in PPO)
  auto advantages = rollout_storage.get_returns().slice(0, 0, -1) - values;

  // Value loss
  auto value_loss = advantages.pow(2).mean();

  // Action loss
  auto action_loss = -(advantages.detach() * action_log_probs).mean();

  // Total loss
  auto loss = (value_loss * value_loss_coef_ +
      action_loss -
      evaluate_result[2] * entropy_coef_);

  // Step optimizer
  optimizer_->zero_grad();
  loss.backward();
  optimizer_->step();

  return {{"Value loss", value_loss.item().toFloat()},
          {"Action loss", action_loss.item().toFloat()},
          {"Entropy", evaluate_result[2].item().toFloat()}};
}
static void learn_pattern(Policy &policy, RolloutStorage &storage, ActorCritic &actor_critic) {
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 5; ++j) {
      auto observation = torch::randint(0, 2, {2, 1});

      std::vector<torch::Tensor> act_result;
      {
        torch::NoGradGuard no_grad;
        act_result = policy->act(observation,
                                 torch::Tensor(),
                                 torch::ones({2, 1}));
      }
      auto actions = act_result[1];

      auto rewards = actions;
      storage.insert(observation,
                     torch::zeros({2, 5}),
                     actions,
                     act_result[2],
                     act_result[0],
                     rewards,
                     torch::ones({2, 1}));
    }

    torch::Tensor next_value;
    {
      torch::NoGradGuard no_grad;
      next_value = policy->get_values(
              storage.get_observations()[-1],
              storage.get_hidden_states()[-1],
              storage.get_masks()[-1])
          .detach();
    }
    storage.compute_returns(next_value, false, 0., 0.9);

    actor_critic.Update(storage,1);
    storage.after_update();
  }
}

static void learn_game(Policy &policy, RolloutStorage &storage, ActorCritic &actor_critic) {
  // The game is: If the action matches the input, give a reward of 1, otherwise -1
  auto observation = torch::randint(0, 2, {2, 1});
  storage.set_first_observation(observation);

  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 5; ++j) {
      std::vector<torch::Tensor> act_result;
      {
        torch::NoGradGuard no_grad;
        act_result = policy->act(observation,
                                 torch::Tensor(),
                                 torch::ones({2, 1}));
      }
      auto actions = act_result[1];

      auto rewards = ((actions == observation.to(torch::kLong)).to(torch::kFloat) * 2) - 1;
      observation = torch::randint(0, 2, {2, 1});
      storage.insert(observation,
                     torch::zeros({2, 5}),
                     actions,
                     act_result[2],
                     act_result[0],
                     rewards,
                     torch::ones({2, 1}));
    }

    torch::Tensor next_value;
    {
      torch::NoGradGuard no_grad;
      next_value = policy->get_values(
              storage.get_observations()[-1],
              storage.get_hidden_states()[-1],
              storage.get_masks()[-1])
          .detach();
    }
    storage.compute_returns(next_value, false, 0.1, 0.9);

    actor_critic.Update(storage,1);
    storage.after_update();
  }
}
}

