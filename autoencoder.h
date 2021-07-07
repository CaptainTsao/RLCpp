//
// Created by transwarp on 2021/6/22.
//

#ifndef _AUTOENCODER_H
#define _AUTOENCODER_H

#include <torch/torch.h>
#include <utility>

struct VAEOutput {
  torch::Tensor reconstruction;
  torch::Tensor mu;
  torch::Tensor log_var;
};

class VAE : public torch::nn::Module {
public:
  VAE(int64_t input_size, int64_t h_dim, int64_t z_dim);

  torch::Tensor decode(torch::Tensor dec_input);

  VAEOutput forward(torch::Tensor enc_input);

  template<class dataloader>
  void train_model(int32_t epochs, torch::Device device, dataloader &data,
                   VAE &model, size_t dataset_size, size_t input_size);

private:
  std::pair<torch::Tensor, torch::Tensor> encode(torch::Tensor enc_input);

  torch::Tensor reparameterize(torch::Tensor mu, torch::Tensor log_var);

  torch::nn::Linear fc1_;
  torch::nn::Linear fc2_;
  torch::nn::Linear fc3_;
  torch::nn::Linear fc4_;
  torch::nn::Linear fc5_;
};

#endif //_AUTOENCODER_H
