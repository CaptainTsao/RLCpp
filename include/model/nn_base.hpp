//
// Created by transwarp on 2021/7/7.
//

#ifndef EXAMPLE_APP_INCLUDE_MODEL_NN_BASE_HPP_
#define EXAMPLE_APP_INCLUDE_MODEL_NN_BASE_HPP_
#include <torch/torch.h>
namespace RLCpp {
class NNBase : public torch::nn::Module {
 private:
  torch::nn::GRU gru_;
  unsigned int hidden_size_;
  bool recurrent_;
 public:
  NNBase(bool recurrent,
         unsigned int hidden_size,
         unsigned int recurrent_input_size);

  virtual std::vector<torch::Tensor> forward(torch::Tensor inputs,
                                             torch::Tensor hxs,
                                             torch::Tensor masks);
  std::vector<torch::Tensor> forward_gru(torch::Tensor inputs,
                                         torch::Tensor hxs,
                                         torch::Tensor masks);
  unsigned int get_hidden_size() const;

  inline int get_output_size() const { return hidden_size_; }
  inline bool is_recurrent() const { return recurrent_; }
};
}
#endif //EXAMPLE_APP_INCLUDE_MODEL_NN_BASE_HPP_
