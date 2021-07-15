//
// Created by transwarp on 2021/7/8.
//

#ifndef EXAMPLE_APP_INCLUDE_DISTRIBUTIONS_DISTRIBUTION_HPP_
#define EXAMPLE_APP_INCLUDE_DISTRIBUTIONS_DISTRIBUTION_HPP_
#include <torch/torch.h>

namespace RLCpp {
class Distribution {
 protected:
  std::vector<int64_t> batch_shape_, event_shape_;
  std::vector<int64_t> extended_shape(const c10::ArrayRef<int64_t> &sample_shape);
 public:
  virtual ~Distribution() = 0;

  virtual torch::Tensor entropy() = 0;
  virtual torch::Tensor log_prob(torch::Tensor value) = 0;
  virtual torch::Tensor sample(const c10::ArrayRef<int64_t> &sample_shape) = 0;
};
inline Distribution::~Distribution() = default;
}
#endif //EXAMPLE_APP_INCLUDE_DISTRIBUTIONS_DISTRIBUTION_HPP_
