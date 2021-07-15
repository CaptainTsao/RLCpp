//
// Created by transwarp on 2021/7/8.
//

#ifndef EXAMPLE_APP_INCLUDE_DISTRIBUTIONS_BERNOULLI_HPP_
#define EXAMPLE_APP_INCLUDE_DISTRIBUTIONS_BERNOULLI_HPP_
#include <c10/util/ArrayRef.h>
#include <torch/torch.h>
#include "distribution.hpp"
namespace RLCpp {
class Bernoulli : public Distribution {
 private:
  torch::Tensor probs_, logits_, param_;
 public:
  Bernoulli(const torch::Tensor *probs, const torch::Tensor *logits);
  torch::Tensor entropy();
  torch::Tensor log_prob(torch::Tensor value);
  torch::Tensor sample(c10::ArrayRef<int64_t> sample_shape = {});

  inline torch::Tensor get_logits() {
    return logits_;
  }
  inline torch::Tensor get_probs() {
    return probs_;
  }
};
}
#endif //EXAMPLE_APP_INCLUDE_DISTRIBUTIONS_BERNOULLI_HPP_
