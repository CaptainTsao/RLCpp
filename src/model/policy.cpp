//
// Created by transwarp on 2021/7/8.
//

#include <utility>

#include "spaces.hpp"
#include "model/policy.hpp"
#include "distributions/categorical.hpp"
#include "model/mlp_base.hpp"
#include "model/output_layers.hpp"
#include "observation_normalizer.hpp"

namespace RLCpp {

PolicyImpl::PolicyImpl(ActionSpace action_space,
                       std::shared_ptr<NNBase> base,
                       bool normalize_observations)
    : action_space_(std::move(action_space)),
      nn_base_(register_module("base", std::move(base))),
      observation_normalizer_(nullptr) {
  int64_t num_outputs = action_space_.shape_[0];
  if (action_space_.type_ == "Discrete") {
    output_layer_ = std::make_shared<CategoricalOutput>(
        nn_base_->get_output_size(), num_outputs
    );
  } else if (action_space_.type_ == "Box") {
    output_layer_ = std::make_shared<NormalOutput>(
        nn_base_->get_output_size(), num_outputs
    );
  } else if (action_space_.type_ == "MultiBinary") {
    output_layer_ = std::make_shared<BernoulliOutput>(
        nn_base_->get_output_size(), num_outputs
    );
  } else {
    throw std::runtime_error("Action space " + action_space_.type_ + " not supported");
  }
  register_module("output", output_layer_);
  if (normalize_observations) {
    auto mlp_base = dynamic_cast<MLPBase *>(nn_base_.get());
    observation_normalizer_ =
        register_module("observation_normalizer", ObservationNormalizer(mlp_base->get_num_inputs()));
  }
}
std::vector<torch::Tensor> PolicyImpl::act(torch::Tensor inputs,
                                           torch::Tensor rnn_hxs,
                                           torch::Tensor masks) const {
  if (observation_normalizer_) {
    inputs = observation_normalizer_->process_observation(inputs);
  }
  auto base_output = nn_base_->forward(inputs, rnn_hxs, masks);
  auto dist = output_layer_->forward(base_output[1]);

  auto action = dist->sample({0});
  auto action_log_probs = dist->log_prob(action);

  if (action_space_.type_ == "Discrete") {
    action = action_log_probs.unsqueeze(-1);
    action_log_probs = action_log_probs.unsqueeze(-1);
  } else {
    action_log_probs = dist->log_prob(action).sum(-1, true);
  }
  return {base_output[0], // value
          action,
          action_log_probs,
          base_output[2]}; // rnn_hxs
}
std::vector<torch::Tensor>
PolicyImpl::evaluate_actions(torch::Tensor inputs,
                             torch::Tensor rnn_hxs,
                             torch::Tensor masks,
                             const torch::Tensor& actions) const {
  if (observation_normalizer_) {
    inputs = observation_normalizer_->process_observation(inputs);
  }
  auto base_output = nn_base_->forward(inputs, std::move(rnn_hxs), masks);
  auto dist = output_layer_->forward(base_output[1]);

  torch::Tensor action_log_probs;
  if (action_space_.type_ == "Discrete") {
    action_log_probs = dist->log_prob(actions.unsqueeze(-1)).
        view({actions.size(0), -1}).
        sum(-1).
        unsqueeze(-1);
  } else {
    action_log_probs = dist->log_prob(actions).
        sum(-1, true);
  }

  auto entropy = dist->entropy().mean();

  return {base_output[0], // value
          action_log_probs,
          entropy,
          base_output[2]}; // rnn_hxs
}
torch::Tensor PolicyImpl::get_probs(torch::Tensor inputs,
                                    torch::Tensor rnn_hxs,
                                    torch::Tensor masks) const {
  if (observation_normalizer_) {
    inputs = observation_normalizer_->process_observation(inputs);
  }

  auto base_output = nn_base_->forward(inputs, rnn_hxs, masks);
  auto dist = output_layer_->forward(base_output[1]);
//  Clang-Tidy: Do not use static_cast to downcast from a base to a derived class; use dynamic_cast instead
  return dynamic_cast<Categorical *>(dist.get())->get_probs();
}
torch::Tensor PolicyImpl::get_values(torch::Tensor inputs,
                                     torch::Tensor rnn_hxs,
                                     torch::Tensor masks) const {
  if (observation_normalizer_) {
    inputs = observation_normalizer_->process_observation(inputs);
  }

  auto base_output = nn_base_->forward(inputs, rnn_hxs, masks);

  return base_output[0];
}
void PolicyImpl::update_observation_normalizer(torch::Tensor observations) {
  assert(!observation_normalizer_.is_empty());
  observation_normalizer_->update(observations);
}
}