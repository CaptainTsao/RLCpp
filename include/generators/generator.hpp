//
// Created by transwarp on 2021/7/7.
//

#ifndef EXAMPLE_APP_INCLUDE_GENERATORS_GENERATOR_HPP_
#define EXAMPLE_APP_INCLUDE_GENERATORS_GENERATOR_HPP_
#include <torch/torch.h>

#include <utility>

namespace RLCpp {
struct MiniBatch {
  torch::Tensor observations_, hidden_states_,
      actions_, value_predictions_,
      returns_, masks_, action_log_probs_, advantages_;
  MiniBatch() = default;
  explicit MiniBatch(torch::Tensor observations,
                     torch::Tensor hidden_states,
                     torch::Tensor actions,
                     torch::Tensor value_predictions,
                     torch::Tensor returns,
                     torch::Tensor masks,
                     torch::Tensor action_log_probs,
                     torch::Tensor advantages) :
      observations_(std::move(observations)),
      hidden_states_(std::move(hidden_states)),
      actions_(std::move(actions)),
      value_predictions_(std::move(value_predictions)),
      returns_(std::move(returns)),
      masks_(std::move(masks)),
      action_log_probs_(std::move(action_log_probs)),
      advantages_(std::move(advantages)) {}
};
class Generator {
  virtual ~Generator() = 0;
  virtual bool done() const = 0;
  virtual MiniBatch next() = 0;
};
inline Generator::~Generator() = default;
}
#endif //EXAMPLE_APP_INCLUDE_GENERATORS_GENERATOR_HPP_
