//
// Created by transwarp on 2021/7/8.
//

#include "../../include/distributions/normal.hpp"

namespace RLCpp {
Normal::Normal(const torch::Tensor loc,
               const torch::Tensor scale) {
  auto broad_casted_tensors = torch::broadcast_tensors({loc, scale});
  this->loc_ = broad_casted_tensors[0];
  this->scale_ = broad_casted_tensors[1];
  batch_shape_ = this->loc_.sizes().vec();
  event_shape_ = {};
}
torch::Tensor Normal::entropy() {
  return (0.5 + 0.5 * std::log(2 * M_PI) + torch::log(scale_)).sum(-1);
}
torch::Tensor Normal::log_prob(torch::Tensor value) {
  auto variance = scale_.pow(2);
  auto log_scale = scale_.log();
  return (-(value - loc_).pow(2) / (2 * variance) - log_scale - std::log(std::sqrt(2 * M_PI)));
}
torch::Tensor Normal::sample(c10::ArrayRef<int64_t> sample_shape) {
  auto shape = extended_shape(sample_shape);
  auto no_grad_guard = torch::NoNamesGuard();
  return at::normal(loc_.expand(shape), scale_.expand(shape));
}
}

