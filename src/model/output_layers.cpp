//
// Created by transwarp on 2021/7/8.
//
//#include <memory>
#include "../../include/model/output_layers.hpp"
#include "../../include/model/model_utils.hpp"
#include "../../include/distributions//distribution.hpp"
#include "../../include/distributions/bernoulli.hpp"
#include "../../include/distributions/categorical.hpp"
#include "../../include/distributions/normal.hpp"

namespace RLCpp {

BernoulliOutput::BernoulliOutput(unsigned int num_inputs, unsigned int num_outputs)
    : linear_(num_inputs, num_inputs) {
  register_module("linear", linear_);
  init_weight(linear_->named_parameters(), 0, 0);
}

std::unique_ptr<Distribution> BernoulliOutput::forward(torch::Tensor x) {
  x = linear_(x);
  return std::make_unique<Bernoulli>(nullptr, &x);
}

CategoricalOutput::CategoricalOutput(unsigned int num_inputs, unsigned int num_outputs)
    : linear_(num_inputs, num_inputs) {
  register_module("linear", linear_);
  init_weight(linear_->named_parameters(), 0.01, 0);
}

std::unique_ptr<Distribution> CategoricalOutput::forward(torch::Tensor x) {
  x = linear_(x);
  return std::make_unique<Categorical>(nullptr, &x);
}

NormalOutput::NormalOutput(unsigned int num_inputs, unsigned int num_outputs)
    : linear_loc_(num_inputs, num_inputs) {
  register_module("linear_loc", linear_loc_);
  scale_log_ = register_parameter("scale_log", torch::zeros({num_outputs}));
  init_weight(linear_loc_->named_parameters(), 1, 0);
}

std::unique_ptr<Distribution> NormalOutput::forward(torch::Tensor x) {
  auto loc = linear_loc_(x);
  auto scale = scale_log_.exp();
  return std::make_unique<Normal>(loc, scale);
}

}