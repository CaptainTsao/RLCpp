//
// Created by transwarp on 2021/7/7.
//
#include <torch/torch.h>
#include "model/model_utils.hpp"

namespace RLCpp {

torch::Tensor orthogonal(torch::Tensor tensor, double gain) {
  torch::NoGradGuard guard;

  AT_ASSERT(tensor.ndimension() >= 2,
            "only tensor with 2 or more dimensions are supported");
  const auto rows = tensor.size(0);
  const auto columns = tensor.numel() / rows;
  auto flattened = torch::randn({rows, columns});

  if (rows < columns)
  {
    flattened.t_();
  }

  // Compute the qr factorization
  torch::Tensor q, r;
  std::tie(q, r) = torch::qr(flattened);
  // Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
  auto d = torch::diag(r, 0);
  auto ph = d.sign();
  q *= ph;

  if (rows < columns)
  {
    q.t_();
  }

  tensor.view_as(q).copy_(q);
  tensor.mul_(gain);

  return tensor;
}

torch::Tensor FlattenImpl::forward(const torch::Tensor &input) {
  return input.view({input.size(0), -1});
}

void init_weight(const torch::OrderedDict<std::string, torch::Tensor> &parameters,
                 double weight_gain,
                 double bias_gain) {
  for (const auto &param: parameters) {
    if (param.value().size(0) != 0) {
      if (param.key().find("bias") != std::string::npos) {
        torch::nn::init::constant_(param.value(), bias_gain);
      } else if (param.key().find("weight") != std::string::npos) {
        orthogonal(param.value(), weight_gain);
      }
    }
  }
}
}