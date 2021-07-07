//
// Created by transwarp on 2021/2/4.
//


#include <torch/torch.h>
#include <iostream>
#include <memory.h>

struct Net : torch::nn::Module {
  Net(int64_t N, int64_t M) :
          linear(register_module("linear", torch::nn::Linear(N, M))) {
    another_bias = register_parameter("b", torch::randn(M));
  }

  torch::Tensor forward(torch::Tensor input) {
    return linear(input) + another_bias;
  }

  torch::nn::Linear linear;
  torch::Tensor another_bias;
};



int main(int argc, char *argv[]) {
  int *p1 = new int(3);
  int *p2 = new int[3];
//  torch::Tensor tensor = torch::eye(3);

//  Net net(4, 5);
//  for (const auto &p : net.named_parameters()) {
//    std::cout << p.key() << ": \n" << p.value() << std::endl;
//  }
//  std::cout << tensor << std::endl;
}