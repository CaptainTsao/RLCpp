//
// Created by transwarp on 2021/7/8.
//

#ifndef OUTPUT_LAYERS_HPP_
#define OUTPUT_LAYERS_HPP_

#include "../distributions/distribution.hpp"
namespace RLCpp {
class OutputLayer : public torch::nn::Module {
 public:
  virtual ~OutputLayer() = default;
  virtual std::unique_ptr<Distribution> forward(torch::Tensor x) = 0;
};

class BernoulliOutput : public OutputLayer {
 private:
  torch::nn::Linear linear_;

 public:
  BernoulliOutput(unsigned int num_inputs, unsigned int num_outputs);

  std::unique_ptr<Distribution> forward(torch::Tensor x);
};
class CategoricalOutput : public OutputLayer {
 private:
  torch::nn::Linear linear_;

 public:
  CategoricalOutput(unsigned int num_inputs, unsigned int num_outputs);

  std::unique_ptr<Distribution> forward(torch::Tensor x);
};

class NormalOutput : public OutputLayer {
 private:
  torch::nn::Linear linear_loc_;
  torch::Tensor scale_log_;

 public:
  NormalOutput(unsigned int num_inputs, unsigned int num_outputs);

  std::unique_ptr<Distribution> forward(torch::Tensor x);
};
}
#endif //EXAMPLE_APP_INCLUDE_MODEL_OUTPUT_LAYERS_HPP_
