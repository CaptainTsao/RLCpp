//
// Created by transwarp on 2021/7/8.
//

#ifndef NORMAL_HPP_
#define NORMAL_HPP_
#include <c10/util/ArrayRef.h>
#include <torch/torch.h>

#include "distribution.hpp"

namespace RLCpp {
class Normal : public Distribution {
 private:
  torch::Tensor loc_, scale_;
 public:
  Normal(const torch::Tensor loc, const torch::Tensor scale);
  torch::Tensor entropy();
  torch::Tensor log_prob(torch::Tensor value);
  torch::Tensor sample(const c10::ArrayRef<int64_t> &sample_shape);

  inline torch::Tensor get_loc() {
    return loc_;
  }
  inline torch::Tensor get_scale() {
    return scale_;
  }
};
}

#endif //EXAMPLE_APP_INCLUDE_DISTRIBUTIONS_NORMAL_HPP_
