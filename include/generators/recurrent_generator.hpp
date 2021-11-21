//
// Created by transwarp on 2021/7/8.
//

#ifndef RECURRENT_GENERATOR_HPP_
#define RECURRENT_GENERATOR_HPP_
#include "generator.hpp"

namespace RLCpp {
class RecurrentGenerator : public Generator
{
 private:
  torch::Tensor observations_, hidden_states_, actions_, value_predictions_,
      returns_, masks_, action_log_probs_, advantages_, indices_;
  int index_, num_envs_per_batch_;

 public:
  RecurrentGenerator(int num_processes,
                     int num_mini_batch,
                     torch::Tensor observations,
                     torch::Tensor hidden_states,
                     torch::Tensor actions,
                     torch::Tensor value_predictions,
                     torch::Tensor returns,
                     torch::Tensor masks,
                     torch::Tensor action_log_probs,
                     torch::Tensor advantages);

  virtual bool done() const;
  virtual MiniBatch next();
};
}
#endif //EXAMPLE_APP_INCLUDE_GENERATORS_RECURRENT_GENERATOR_HPP_
