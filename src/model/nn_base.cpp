//
// Created by transwarp on 2021/7/8.
//

#include "../../include/model/nn_base.hpp"
#include "../../include/model/model_utils.hpp"

namespace RLCpp {
NNBase::NNBase(bool recurrent,
               unsigned int hidden_size,
               unsigned int recurrent_input_size) :
    gru_(nullptr),
    hidden_size_(hidden_size),
    recurrent_(recurrent) {
  // Init GRU
  if (recurrent_) {
    gru_ = torch::nn::GRU(torch::nn::GRUOptions(recurrent_input_size, hidden_size));
    register_module("gru", gru_);
    // Init weights
    init_weight(gru_->named_parameters(), 1, 0);
  }
}
std::vector<torch::Tensor>
NNBase::forward(torch::Tensor inputs,
                torch::Tensor hxs,
                torch::Tensor masks) {
  return std::vector<torch::Tensor>();
}

std::vector<torch::Tensor>
NNBase::forward_gru(torch::Tensor inputs,
                    torch::Tensor hxs,
                    torch::Tensor masks) {
  if (inputs.size(0) == hxs.size(0)) {
    auto gru_output = gru_->forward(inputs.unsqueeze(0),
                                    (hxs * masks).unsqueeze(0));
    return {gru_output.output, gru_output.hidden};
  } else {
    // x is a (timesteps, agents, -1) tensor that has been flattened to
    // (timesteps * agents, -1)
    auto agents = hxs.size(0);
    auto timesteps = inputs.size(0) / agents;

    // Unflatten
    inputs = inputs.view({timesteps, agents, inputs.size(1)});

    // Same for masks
    masks = masks.view({timesteps, agents});

    // Figure out which steps in the sequence have a zero for any agent
    // We assume the first timestep has a zero in it
    auto has_zeros = (masks.narrow(0, 1, masks.size(0) - 1) == 0)
        .any(-1)
        .nonzero()
        .squeeze();

    // +1 to correct the masks[1:]
    has_zeros += 1;

    // Add t=0 and t=timesteps to the list
    // has_zeros = [0] + has_zeros + [timesteps]
    has_zeros = has_zeros.contiguous().to(torch::kInt);
    std::vector<int> has_zeros_vec(
        has_zeros.data_ptr<int>(),
        has_zeros.data_ptr<int>() + has_zeros.numel());
    has_zeros_vec.insert(has_zeros_vec.begin(), {0});
    has_zeros_vec.push_back(timesteps);

    hxs = hxs.unsqueeze(0);
    std::vector<torch::Tensor> outputs;
    for (unsigned int i = 0; i < has_zeros_vec.size() - 1; ++i) {
      // We can now process long runs of timesteps without dones in them in
      // one go
      auto start_idx = has_zeros_vec[i];
      auto end_idx = has_zeros_vec[i + 1];

      auto gru_output = gru_(inputs.index({torch::arange(start_idx,
                                                         end_idx,
                                                         torch::TensorOptions(torch::kLong))}),
                             hxs * masks[start_idx].view({1, -1, 1}));

      outputs.push_back(gru_output.output);
    }

    // x is a (timesteps, agents, -1) tensor
    inputs = torch::cat(outputs, 0).squeeze(0);
    inputs = inputs.view({timesteps * agents, -1});
    hxs = hxs.squeeze(0);

    return {inputs, hxs};
  }
}
unsigned int NNBase::get_hidden_size() const {
  return 0;
}
}


