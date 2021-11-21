//
// Created by transwarp on 2021/7/8.
//

#include <utility>

#include "generators/generator.hpp"
#include "generators/recurrent_generator.hpp"

namespace RLCpp {

torch::Tensor flatten_helper(int64_t time_steps, int processes, torch::Tensor tensor) {
  auto tensor_shape = tensor.sizes().vec();
  tensor_shape.erase(tensor_shape.begin());
  tensor_shape[0] = time_steps * processes;
  return tensor.view(tensor_shape);
}
RecurrentGenerator::RecurrentGenerator(int num_processes,
                                       int num_mini_batch,
                                       torch::Tensor observations,
                                       torch::Tensor hidden_states,
                                       torch::Tensor actions,
                                       torch::Tensor value_predictions,
                                       torch::Tensor returns,
                                       torch::Tensor masks,
                                       torch::Tensor action_log_probs,
                                       torch::Tensor advantages)
    : observations_(std::move(observations)),
      hidden_states_(std::move(hidden_states)),
      actions_(std::move(actions)),
      value_predictions_(std::move(value_predictions)),
      returns_(std::move(returns)),
      masks_(std::move(masks)),
      action_log_probs_(std::move(action_log_probs)),
      advantages_(std::move(advantages)),
      indices_(torch::randperm(num_processes, torch::TensorOptions(torch::kLong))),
      index_(0),
      num_envs_per_batch_(num_processes / num_mini_batch) {}

bool RecurrentGenerator::done() const {
  return index_ >= indices_.size(0);
}
MiniBatch RecurrentGenerator::next() {
  if (index_ >= indices_.size(0)) {
    throw std::runtime_error("No mini-batches left in generator.");
  }
  MiniBatch mini_batch;
  // Fill minibatch with tensors of shape (timestep, process, *whatever)
  // Except hidden states, that is just (process, *whatever)
  int64_t env_index = indices_[index_].item().toLong();
  mini_batch.observations_ = observations_
      .narrow(0, 0, observations_.size(0) - 1)
      .narrow(1, env_index, num_envs_per_batch_);
  mini_batch.hidden_states_ = hidden_states_[0]
      .narrow(0, env_index, num_envs_per_batch_)
      .view({num_envs_per_batch_, -1});
  mini_batch.actions_ = actions_.narrow(1, env_index, num_envs_per_batch_);
  mini_batch.value_predictions_ = value_predictions_
      .narrow(0, 0, value_predictions_.size(0) - 1)
      .narrow(1, env_index, num_envs_per_batch_);
  mini_batch.returns_ = returns_.narrow(0, 0, returns_.size(0) - 1)
      .narrow(1, env_index, num_envs_per_batch_);
  mini_batch.masks_ = masks_.narrow(0, 0, masks_.size(0) - 1)
      .narrow(1, env_index, num_envs_per_batch_);
  mini_batch.action_log_probs_ = action_log_probs_.narrow(1, env_index,
                                                          num_envs_per_batch_);
  mini_batch.advantages_ = advantages_.narrow(1, env_index, num_envs_per_batch_);

  // Flatten tensors to (timestep * process, *whatever)
  int64_t num_time_steps = mini_batch.observations_.size(0);
  int num_processes = num_envs_per_batch_;
  mini_batch.observations_ = flatten_helper(num_time_steps, num_processes,
                                            mini_batch.observations_);
  mini_batch.actions_ = flatten_helper(num_time_steps, num_processes,
                                       mini_batch.actions_);
  mini_batch.value_predictions_ = flatten_helper(num_time_steps, num_processes,
                                                 mini_batch.value_predictions_);
  mini_batch.returns_ = flatten_helper(num_time_steps, num_processes,
                                       mini_batch.returns_);
  mini_batch.masks_ = flatten_helper(num_time_steps, num_processes,
                                     mini_batch.masks_);
  mini_batch.action_log_probs_ = flatten_helper(num_time_steps, num_processes,
                                                mini_batch.action_log_probs_);
  mini_batch.advantages_ = flatten_helper(num_time_steps, num_processes,
                                          mini_batch.advantages_);

  index_++;

  return mini_batch;
}
}