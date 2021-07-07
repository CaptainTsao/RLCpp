//
// Created by transwarp on 2021/7/7.
//

#ifndef EXAMPLE_APP_INCLUDE_MODEL_MLP_BASE_HPP_
#define EXAMPLE_APP_INCLUDE_MODEL_MLP_BASE_HPP_

#include <torch/torch.h>
#include "nn_base.hpp"

namespace RLCpp {
class MLPBase : public NNBase {
 private:
  torch::nn::Sequential actor_;
  torch::nn::Sequential critic_;
  torch::nn::Linear critic_linear_;
  unsigned int num_inputs;
 public:
  MLPBase(unsigned int num_inputs,
          bool recurrent = false,
          unsigned int hidden_size = 64);
  std::vector<torch::Tensor> forward(torch::Tensor inputs,
                                     torch::Tensor hxs,
                                     torch::Tensor masks) override;
  inline unsigned int get_num_inputs() const { return num_inputs; }

};
}
#endif //EXAMPLE_APP_INCLUDE_MODEL_MLP_BASE_HPP_
