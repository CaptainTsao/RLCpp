//
// Created by transwarp on 2021/7/8.
//
#include "../../include/model/mlp_base.hpp"
#include "../../include/model/model_utils.hpp"

namespace RLCpp {

MLPBase::MLPBase(unsigned int num_inputs,
                 bool recurrent,
                 unsigned int hidden_size) :
    NNBase(recurrent, num_inputs, hidden_size),
    actor_(nullptr),
    critic_(nullptr),
    critic_linear_(nullptr),
    num_inputs(num_inputs) {
  if (recurrent) {
    // If using a recurrent architecture, the inputs are first processed through
    // a GRU layer, so the actor and critic parts of the network take the hidden
    // size as their input size.
    num_inputs = hidden_size;
  }
  actor_ = torch::nn::Sequential(torch::nn::Linear(num_inputs, hidden_size),
                                 torch::nn::Functional(torch::tanh),
                                 torch::nn::Linear(hidden_size, hidden_size),
                                 torch::nn::Functional(torch::tanh));
  critic_ = torch::nn::Sequential(torch::nn::Linear(num_inputs, hidden_size),
                                  torch::nn::Functional(torch::tanh),
                                  torch::nn::Linear(hidden_size, hidden_size),
                                  torch::nn::Functional(torch::tanh));
  critic_linear_ = torch::nn::Linear(hidden_size, 1);

  register_module("actor", actor_);
  register_module("critic", critic_);
  register_module("critic_linear", critic_linear_);

  init_weight(actor_->named_parameters(), sqrt(2.), 0);
  init_weight(critic_->named_parameters(), sqrt(2.), 0);
  init_weight(critic_linear_->named_parameters(), sqrt(2.0), 0);

  train();
}
std::vector<torch::Tensor> MLPBase::forward(torch::Tensor inputs,
                                            torch::Tensor hxs,
                                            torch::Tensor masks) {
  auto x = inputs;

  if (is_recurrent()) {
    auto gru_output = forward_gru(x, hxs, masks);
    x = gru_output[0];
    hxs = gru_output[1];
  }
  auto hidden_critic = critic_->forward(x);
  auto hidden_actor = actor_->forward(x);

  return {critic_linear_->forward(hidden_critic), hidden_actor, hxs};
}
}