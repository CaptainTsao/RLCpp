//
// Created by transwarp on 2021/7/8.
//

#include "distributions/categorical.hpp"
#include "third_party/doctest.hpp"

namespace RLCpp {
Categorical::Categorical(const torch::Tensor *probs, const torch::Tensor *logits) {
  if ((probs == nullptr) == (logits == nullptr)) {
    throw std::runtime_error("Either probs or logits is required, but not both");
  }
  if (probs != nullptr) {
    if (probs->dim() < 1) {
      throw std::runtime_error("Probabilities tensor must have at least one dimension");
    }
    this->probs_ = *probs / probs->sum(-1, true);
    this->probs_ = this->probs_.clamp(1.21e-7, 1. - 1.21e-7);
    this->logits_ = torch::log(this->probs_);
  } else {
    if (logits_.dim() < 1) {
      throw std::runtime_error("Logits tensor must have at least one dimension");
    }
    this->logits_ = *logits - logits->logsumexp(-1, true);
    this->probs_ = torch::softmax(this->logits_, -1);
  }
  param_ = probs != nullptr ? *probs : *logits;
  num_events_ = param_.size(-1);
  if (param_.dim() > 1) {
    batch_shape_ = param_.sizes().vec();
    batch_shape_.resize(batch_shape_.size() - 1);
  }
}
torch::Tensor Categorical::entropy() {
  auto p_log_p = logits_ * probs_;
  return -p_log_p.sum(-1);
}
torch::Tensor Categorical::log_prob(torch::Tensor value) {
  value = value.to(torch::kLong).unsqueeze(-1);
  auto broad_casted_tensors = torch::broadcast_tensors({value, logits_});
  value = broad_casted_tensors[0];
  value = value.narrow(-1, 0, 1);
  return broad_casted_tensors[1].gather(-1, value).squeeze(-1);;
}
torch::Tensor Categorical::sample(const c10::ArrayRef<int64_t> &sample_shape) {
  auto ext_sample_shape = extended_shape(sample_shape);
  auto param_shape = ext_sample_shape;
  param_shape.insert(param_shape.end(), {num_events_});
  auto exp_probs = probs_.expand(param_shape);
  torch::Tensor probs_2d;
  if (probs_.dim() == 1 || probs_.size(0) == 1) {
    probs_2d = exp_probs.view({-1, num_events_});
  } else {
    probs_2d = exp_probs.contiguous().view({-1, num_events_});
  }
  auto sample_2d = torch::multinomial(probs_2d, 1, true);
  return sample_2d.contiguous().view(ext_sample_shape);
}

TEST_CASE ("Categorical")
{
      SUBCASE("Throw when provided both probs and logits") {
    auto tensor = torch::Tensor();
        CHECK_THROWS(Categorical(&tensor, &tensor));
  }
}
}

