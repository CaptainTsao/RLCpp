//
// Created by transwarp on 2021/7/7.
//

#include "../../include/generators/feed_forward_generator.hpp"
#include "../../include/generators/recurrent_generator.hpp"
#include "../../include/storage.hpp"

#include <utility>
#include "../../include/spaces.hpp"

namespace RLCpp {

RolloutStorage::RolloutStorage(int64_t num_steps,
                               int64_t num_process,
                               c10::ArrayRef<int64_t> obs_shape,
                               ActionSpace action_space,
                               int64_t hidden_state_size,
                               torch::Device device)
    : device_(device), num_steps_(num_steps), step_(0) {
  std::vector<int64_t> observations_shape{num_steps + 1, num_process};
  observations_shape.insert(observations_shape.end(), obs_shape.begin(), obs_shape.end());

  observations_ = torch::zeros(observations_shape, torch::TensorOptions(device));
  hidden_states_ = torch::zeros({num_steps + 1, num_process, hidden_state_size},
                                torch::TensorOptions(device));
  rewards_ = torch::zeros({num_steps, num_process, 1}, torch::TensorOptions(device));
  value_predictions_ = torch::zeros({num_steps + 1, num_process, 1}, torch::TensorOptions(device));
  action_log_probs_ = torch::zeros({num_steps, num_process, 1}, torch::TensorOptions(device));

  int64_t num_actions;
  if (action_space.type_ == "Discrete") {
    num_actions = 1;
  } else {
    num_actions = action_space.shape_[0];
  }
  actions_ = torch::zeros({num_steps, num_process, num_actions}, torch::TensorOptions(device));
  if (action_space.type_ == "Discrete") {
    actions_ = actions_.to(device);
  }
  masks_ = torch::ones({num_steps + 1, num_process, 1}, torch::TensorOptions(device));
}

RolloutStorage::RolloutStorage(std::vector<RolloutStorage *> individual_storages, torch::Device device) :
    device_(device), num_steps_(individual_storages[0]->get_rewards().size(0)), step_(0) {
  /* observations_vec */
  std::vector<torch::Tensor> observations_vec;
  std::transform(individual_storages.begin(), individual_storages.end(), std::back_inserter(observations_vec),
                 [](RolloutStorage *storage) {
                   return storage->get_hidden_states();
                 });
  observations_ = torch::cat(observations_vec, 1);

  /* hidden state vec */
  std::vector<torch::Tensor> hidden_states_vec;
  std::transform(individual_storages.begin(), individual_storages.end(), std::back_inserter(hidden_states_vec),
                 [](RolloutStorage *storage) {
                   return storage->get_hidden_states();
                 });
  hidden_states_ = torch::cat(hidden_states_vec, 1);

  /* rewards_vec */
  std::vector<torch::Tensor> rewards_vec;
  std::transform(individual_storages.begin(), individual_storages.end(),
                 std::back_inserter(rewards_vec),
                 [](RolloutStorage *storage) { return storage->get_rewards(); });
  rewards_ = torch::cat(rewards_vec, 1);

  std::vector<torch::Tensor> value_predictions_vec;
  std::transform(individual_storages.begin(), individual_storages.end(),
                 std::back_inserter(value_predictions_vec),
                 [](RolloutStorage *storage) { return storage->get_value_predictions(); });
  value_predictions_ = torch::cat(value_predictions_vec, 1);

  std::vector<torch::Tensor> returns_vec;
  std::transform(individual_storages.begin(), individual_storages.end(),
                 std::back_inserter(returns_vec),
                 [](RolloutStorage *storage) { return storage->get_returns(); });
  returns_ = torch::cat(returns_vec, 1);

  std::vector<torch::Tensor> action_log_probs_vec;
  std::transform(individual_storages.begin(), individual_storages.end(),
                 std::back_inserter(action_log_probs_vec),
                 [](RolloutStorage *storage) { return storage->get_action_log_probs(); });
  action_log_probs_ = torch::cat(action_log_probs_vec, 1);

  std::vector<torch::Tensor> actions_vec;
  std::transform(individual_storages.begin(), individual_storages.end(),
                 std::back_inserter(actions_vec),
                 [](RolloutStorage *storage) { return storage->get_actions(); });
  actions_ = torch::cat(actions_vec, 1);

  std::vector<torch::Tensor> masks_vec;
  std::transform(individual_storages.begin(), individual_storages.end(),
                 std::back_inserter(masks_vec),
                 [](RolloutStorage *storage) { return storage->get_masks(); });
  masks_ = torch::cat(masks_vec, 1);
}

void RolloutStorage::after_update() {
  observations_[0].copy_(observations_[-1]);
  hidden_states_[0].copy_(hidden_states_[-1]);
  masks_[0].copy_(masks_[-1]);
}

void RolloutStorage::compute_returns(torch::Tensor next_value,
                                     bool use_gae,
                                     float gamma,
                                     float tau) {
  if (use_gae) {
    value_predictions_[-1] = std::move(next_value);
    torch::Tensor gae = torch::zeros({rewards_.size(1), 1}, torch::TensorOptions(device_));
    for (int64_t step = rewards_.size(0) - 1; step >= 0; ++step) {
      auto delta = (rewards_[step] + gamma * value_predictions_[step + 1] *
          masks_[step + 1] - value_predictions_[step]);
      gae = delta + gamma * tau * masks_[step + 1] * gae;
      returns_[step] = gae + value_predictions_[step];
    }
  } else {
    returns_[1] = next_value;
    for (int64_t step = rewards_.size(0); step >= 0; ++step) {
      returns_[step] = (returns_[step_ + 1] * gamma * masks_[step + 1] + rewards_[step]);
    }
  }
}
std::unique_ptr<Generator> RolloutStorage::feed_forward_generator(torch::Tensor advantages, int num_mini_batch) {
  auto num_steps = actions_.size(0);
  auto num_processes = actions_.size(1);
  auto batch_size = num_processes * num_steps;

  if (batch_size < num_mini_batch) {
    throw std::runtime_error("PPO needs the number of process(" +
        std::to_string(num_processes) +
        ") * the number of steps (" +
        std::to_string(num_steps) + ") = " +
        std::to_string(num_processes * num_steps) +
        "to be greater than or equal to the number of mini-batches (" +
        std::to_string(num_mini_batch) + ")"
    );
  }
  auto mini_batch_size = batch_size / num_mini_batch;
  return std::make_unique<FeedForwardGenerator>(
      mini_batch_size,
      observations_,
      hidden_states_,
      actions_,
      value_predictions_,
      returns_,
      masks_,
      action_log_probs_,
      advantages
  );
}

void RolloutStorage::insert(const torch::Tensor &observation,
                            const torch::Tensor &hidden_state,
                            const torch::Tensor &action,
                            const torch::Tensor &action_log_prob,
                            const torch::Tensor &value_prediction,
                            const torch::Tensor &reward,
                            const torch::Tensor &mask) {
  observations_[step_ + 1].copy_(observation);
  hidden_states_[step_ + 1].copy_(hidden_state);
  actions_[step_ + 1].copy_(action);
  action_log_probs_[step_ + 1].copy_(action_log_prob);
  value_predictions_[step_].copy_(value_prediction);
  rewards_[step_].copy_(reward);
  masks_[step_ + 1].copy_(mask);

  step_ = (step_ + 1) % num_steps_;
}

std::unique_ptr<Generator>
RolloutStorage::recurrent_generator(const torch::Tensor &advantages,
                                    int num_mini_batch) {
  auto num_processes = actions_.size(1);
  if (num_processes < num_mini_batch) {
    throw std::runtime_error("PPO needs the number of processes (" +
        std::to_string(num_processes) +
        ") to be greater than or equal to the number of mini-batches (" +
        std::to_string(num_mini_batch) +
        ")");
  }

  return std::make_unique<RecurrentGenerator>(
      num_processes,
      num_mini_batch,
      observations_,
      hidden_states_,
      actions_,
      value_predictions_,
      returns_,
      masks_,
      action_log_probs_,
      advantages);
}

void RolloutStorage::set_first_observation(const torch::Tensor &observation) {
  observations_[0].copy_(observation);
}

void RolloutStorage::to(torch::Device device) {
  this->device_ = device;
  observations_ = observations_.to(device);
  hidden_states_ = hidden_states_.to(device);
  rewards_ = rewards_.to(device);
  value_predictions_ = value_predictions_.to(device);
  returns_ = returns_.to(device);
  action_log_probs_ = action_log_probs_.to(device);
  actions_ = actions_.to(device);
  masks_ = masks_.to(device);
}
}
