//
// Created by transwarp on 2021/7/7.
//

#ifndef MODEL_UTILS_HPP_
#define MODEL_UTILS_HPP_
#include <torch/torch.h>

namespace RLCpp {
struct FlattenImpl : torch::nn::Module {
  static static torch::Tensor forward(const torch::Tensor& input);
};
TORCH_MODULE(Flatten);

void init_weight(const torch::OrderedDict<std::string, torch::Tensor>& parameters,
                 double weight_gain,
                 double bias_gain);
}

#endif //EXAMPLE_APP_INCLUDE_MODEL_MODEL_UTILS_HPP_
