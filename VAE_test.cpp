//
// Created by transwarp on 2021/6/22.
//

#include <iostream>
#include <torch/torch.h>
#include "autoencoder.h"



int main() {
  torch::manual_seed(1234);
  torch::DeviceType device_type = torch::kCPU;
  torch::Device device(device_type);

  int64_t hid = 400;
  int64_t dec_input = 20;
  int64_t input_size = 28 * 28;
  int64_t batch = 50;

  VAE model(input_size, hid, dec_input);

  auto dataset = torch::data::datasets::MNIST("").
          map(torch::data::transforms::Normalize<>(0.1307, 0.3081));
  size_t dt_size = dataset.size().value();
  auto train_loader = torch::data::make_data_loader<torch::data::samplers::StreamSampler>
          (std::move(dataset), batch);
//  std::setprecision(16);
  model.train(true);
  model.train_model(100, device, train_loader, model, dt_size, input_size);

  return 0;
}