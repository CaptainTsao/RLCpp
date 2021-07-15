//
// Created by transwarp on 2021/7/8.
//

#ifndef EXAMPLE_APP_INCLUDE_MODEL_POLICY_HPP_
#define EXAMPLE_APP_INCLUDE_MODEL_POLICY_HPP_
#include <torch/torch.h>

#include "nn_base.hpp"
#include "output_layers.hpp"
#include "../spaces.hpp"
#include "../observation_normalizer.hpp"

namespace RLCpp {
class PolicyImpl : public torch::nn::Module {
 private:
  ActionSpace action_space_;
  std::shared_ptr<NNBase> nn_base_;
  ObservationNormalizer observation_normalizer_;
  std::shared_ptr<OutputLayer> output_layer_;
  std::vector<torch::Tensor> forward_gru(torch::Tensor x,
                                         torch::Tensor hxs,
                                         torch::Tensor masks);

 public:
  PolicyImpl(ActionSpace action_space,
             std::shared_ptr<NNBase> base,
             bool normalize_observations = false);

  std::vector<torch::Tensor> act(torch::Tensor inputs,
                                 torch::Tensor rnn_hxs,
                                 torch::Tensor masks) const;
  std::vector<torch::Tensor> evaluate_actions(torch::Tensor inputs,
                                              torch::Tensor rnn_hxs,
                                              torch::Tensor masks,
                                              const torch::Tensor& actions) const;
  torch::Tensor get_probs(torch::Tensor inputs,
                          torch::Tensor rnn_hxs,
                          torch::Tensor masks) const;
  torch::Tensor get_values(torch::Tensor inputs,
                           torch::Tensor rnn_hxs,
                           torch::Tensor masks) const;
  void update_observation_normalizer(torch::Tensor observations);

  inline bool is_recurrent() const {
    return nn_base_->is_recurrent();
  }
  inline unsigned int get_hidden_size() const {
    return nn_base_->get_hidden_size();
  }
  inline bool using_observation_normalizer() const {
    return !observation_normalizer_.is_empty();
  }
};
TORCH_MODULE(Policy);
}
#endif //EXAMPLE_APP_INCLUDE_MODEL_POLICY_HPP_
