//
// Created by transwarp on 2021/7/8.
//

#include <utility>

#include "generators/feed_forward_generator.hpp"

namespace RLCpp {

FeedForwardGenerator::FeedForwardGenerator(int mini_batch_size,
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
      index_(0) {
  int64_t batch_size = advantages_.numel();
  indices_ = torch::randperm(batch_size,
                             torch::TensorOptions(torch::kLong))
      .view({-1, mini_batch_size});
}

MiniBatch FeedForwardGenerator::next() {
  if (index_ >= indices_.size(0)) {
    throw std::runtime_error("No mini-batches left in generator.");
  }

  MiniBatch mini_batch;

  int64_t time_steps = observations_.size(0) - 1;

  auto observations_shape = observations_.sizes().vec();
  observations_shape.erase(observations_shape.begin());
  observations_shape[0] = -1;
  mini_batch.observations_ = observations_.narrow(0, 0, time_steps)
      .view(observations_shape)
      .index(indices_[index_]);
  mini_batch.hidden_states_ = hidden_states_.narrow(0, 0, time_steps)
      .view({-1, hidden_states_.size(-1)})
      .index(indices_[index_]);
  mini_batch.actions_ = actions_.view({-1, actions_.size(-1)})
      .index(indices_[index_]);
  mini_batch.value_predictions_ = value_predictions_.narrow(0, 0, time_steps)
      .view({-1, 1})
      .index(indices_[index_]);
  mini_batch.returns_ = returns_.narrow(0, 0, time_steps)
      .view({-1, 1})
      .index(indices_[index_]);
  mini_batch.masks_ = masks_.narrow(0, 0, time_steps)
      .view({-1, 1})
      .index(indices_[index_]);
  mini_batch.action_log_probs_ = action_log_probs_.view({-1, 1})
      .index(indices_[index_]);
  mini_batch.advantages_ = advantages_.view({-1, 1})
      .index(indices_[index_]);

  index_++;
  return mini_batch;
}
bool FeedForwardGenerator::done() const {
  return index_ >= indices_.size(0);
}
}