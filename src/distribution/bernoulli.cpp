//
// Created by transwarp on 2021/7/8.
//

#include "distributions/bernoulli.hpp"

namespace RLCpp {
Bernoulli::Bernoulli(const torch::Tensor *probs,
                     const torch::Tensor *logits) {
  if ((probs == nullptr) == (logits == nullptr)) {
    throw std::runtime_error("Either probs or logits is required, but not both");
  }
  if (probs != nullptr) {
    if (probs->dim() < 1) {
      throw std::runtime_error("Probabilities tensor must have at least one dimension");
    }
    this->probs_ = *probs;
    auto clamped_probs = this->probs_.clamp(1.21e-7, 1. - 1.21e-7);
    this->logits_ = torch::log(clamped_probs) - torch::log1p(-clamped_probs);
  } else {
    if (logits_.dim() < 1) {
      throw std::runtime_error("Logits tensor must have at least one dimension");
    }
    this->logits_ = *logits;
    this->probs_ = torch::sigmoid(*logits);
  }
  param_ = probs != nullptr ? *probs : *logits;
  batch_shape_ = param_.sizes().vec();
}
torch::Tensor Bernoulli::entropy() {
  return torch::binary_cross_entropy_with_logits(logits_,
                                                 probs_,
                                                 torch::Tensor(),
                                                 torch::Tensor(),
                                                 torch::Reduction::None);
}
torch::Tensor Bernoulli::log_prob(torch::Tensor value) {
  auto broadcasted_tensors = torch::broadcast_tensors({logits_, value});
  return -torch::binary_cross_entropy_with_logits(broadcasted_tensors[0],
                                                  broadcasted_tensors[1],
                                                  torch::Tensor(),
                                                  torch::Tensor(),
                                                  torch::Reduction::None);
}
torch::Tensor Bernoulli::sample(const c10::ArrayRef<int64_t> &sample_shape) {
  auto ext_sample_shape = extended_shape(sample_shape);
  torch::NoGradGuard no_grad_guard;
  return torch::bernoulli(probs_.expand(ext_sample_shape));
}

}


