//
// Created by transwarp on 2021/6/22.
//

#include "autoencoder.h"
#include <cstddef>
#include <iostream>

using namespace std;

VAE::VAE(int64_t input_size, int64_t h_dim, int64_t z_dim) :
        fc1_(input_size, z_dim), fc2_(h_dim, z_dim),
        fc3_(h_dim, z_dim), fc4_(h_dim, z_dim),
        fc5_(h_dim, h_dim) {
  register_module("fc1", fc1_);
  register_module("fc2", fc2_);
  register_module("fc3", fc3_);
  register_module("fc4", fc4_);
  register_module("fc5", fc5_);
}

VAEOutput VAE::forward(torch::Tensor enc_input) {
  auto encoded = encode(enc_input);
  auto mu = encoded.first;
  auto log_var = encoded.second;
  auto z = reparameterize(mu, log_var);
  auto reconstructed = decode(z);
  return {reconstructed, mu, log_var};
}

template<class dataloader>
void VAE::train_model(int32_t epochs, torch::Device device, dataloader &data,
                      VAE &model, size_t dataset_size, size_t input_size) {
  model.train(true);
  model.to(device);
  size_t batch_idx = 0;
  torch::optim::Adam adam(model.parameters(), torch::optim::AdamOptions(0.1));
  for (auto &batch: data) {
    // clear the optimizer parameters
    adam.zero_grad();
    torch::Tensor data = batch.data.to(device).reshape({-1, input_size});
    auto output = model.forward(data);
    auto reconstruction_loss = torch::nn::functional::binary_cross_entropy(
            output.reconstruction, data,
            torch::nn::functional::BinaryCrossEntropyFuncOptions().reduction(torch::kSum)
    );
    auto kl_divergence = -0.5 * torch::sum(1 + output.log_var - output.mu.pow(2) - output.log_var.exp());
    auto loss = reconstruction_loss + kl_divergence;
//    AT_ASSERT(!std::nan(loss.template item<float>));
    loss.backward();
    adam.step();
  }
}

torch::Tensor VAE::reparameterize(torch::Tensor mu, torch::Tensor log_var) {
  if (is_training()) {
    torch::Tensor scale = log_var.div(2).exp();
    torch::Tensor eps = torch::rand_like(scale);
    return eps.mul(scale).add(mu);
  } else {
    return mu;
  }
}

std::pair<torch::Tensor, torch::Tensor> VAE::encode(torch::Tensor enc_input) {
  auto hid = torch::nn::functional::relu(fc1_->forward(enc_input));
  return {fc2_->forward(hid), fc3_->forward(hid)};
}

torch::Tensor VAE::decode(torch::Tensor dec_input) {
  torch::Tensor hid = torch::nn::functional::relu(fc4_->forward(dec_input));
  return torch::sigmoid(fc5_->forward(hid));
}