//
// Created by transwarp on 2021/7/7.
//

#include "include/storage.hpp"

RLCpp::RolloutStorage::RolloutStorage(int64_t num_steps,
                                      int64_t num_process,
                                      c10::ArrayRef<int64_t> obs_shape,
                                      ActionSpace action_space,
                                      int64_t hidden_state_size,
                                      torch::Device device) {

}
RLCpp::RolloutStorage::RolloutStorage(std::vector<RolloutStorage *> individual_storage, torch::Device device) {

}
void RLCpp::RolloutStorage::after_update() {

}
void RLCpp::RolloutStorage::compute_returns(torch::Tensor next_value, bool use_gae, float gamma, float tau) {

}
void RLCpp::RolloutStorage::set_first_observation(torch::Tensor observation) {

}
std::unique_ptr<Generator> RLCpp::RolloutStorage::feed_forward_generator(torch::Tensor advantages, int num_mini_batch) {
  return std::unique_ptr<Generator>();
}

