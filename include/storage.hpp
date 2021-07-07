//
// Created by transwarp on 2021/7/7.
//

#ifndef EXAMPLE_APP_INCLUDE_STORAGE_HPP_
#define EXAMPLE_APP_INCLUDE_STORAGE_HPP_
#include <c10/util/ArrayRef.h>
#include <torch/torch.h>

#include <utility>
#include "generators/generator.hpp"
#include "spaces.hpp"
namespace RLCpp {
class RolloutStorage {
 private:
  torch::Tensor observations_, hidden_states_,
      rewards_, value_predictions_,
      returns_, masks_, action_log_probs_, actions_;
  torch::Device device_;
  int64_t num_steps_;
  int64_t step_;

 public:
  RolloutStorage(int64_t num_steps, int64_t num_process,
                 c10::ArrayRef<int64_t> obs_shape,
                 ActionSpace action_space,
                 int64_t hidden_state_size,
                 torch::Device device);
  RolloutStorage(std::vector<RolloutStorage *> individual_storage,
                 torch::Device device);
  void after_update();
  void compute_returns(torch::Tensor next_value,
                       bool use_gae, float gamma, float tau);
  std::unique_ptr<Generator> feed_forward_generator(torch::Tensor advantages,
                                                    int num_mini_batch);
  void set_first_observation(torch::Tensor observation);
  void to(torch::Device device);

  inline const torch::Tensor &get_actions() const {
    return actions_;
  }
  inline const torch::Tensor &get_action_log_probs() const {
    return action_log_probs_;
  }
  inline const torch::Tensor &get_hidden_states() const {
    return hidden_states_;
  }
  inline const torch::Tensor &get_masks() const {
    return masks_;
  }
  inline const torch::Tensor &get_observations() const {
    return observations_;
  }
  inline const torch::Tensor &get_returns() const {
    return returns_;
  }
  inline const torch::Tensor &get_rewards() const {
    return rewards_;
  }
  inline const torch::Tensor &get_value_predictions() const {
    return value_predictions_;
  }
  inline void set_actions(torch::Tensor actions) { this->actions_ = std::move(actions); }
  inline void set_action_log_probs(torch::Tensor action_log_probs) {
    this->action_log_probs_ = std::move(action_log_probs);
  }
  inline void set_hidden_states(torch::Tensor hidden_states) {
    this->hidden_states_ = std::move(hidden_states);
  }
  inline void set_masks(torch::Tensor masks) {
    this->masks_ = std::move(masks);
  }
  inline void set_observations(torch::Tensor observations) {
    this->observations_ = std::move(observations);
  }
  inline void set_returns(torch::Tensor returns) {
    this->returns_ = std::move(returns);
  }
  inline void set_rewards(torch::Tensor rewards) {
    this->rewards_ = std::move(rewards);
  }
  inline void set_value_predictions(torch::Tensor value_predictions) {
    this->value_predictions_ = std::move(value_predictions);
  }
};
}
#endif //EXAMPLE_APP_INCLUDE_STORAGE_HPP_
