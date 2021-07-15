//
// Created by transwarp on 2021/7/8.
//

#ifndef EXAMPLE_APP_INCLUDE_MODEL_OBSERVATION_NORMALIZER_HPP_
#define EXAMPLE_APP_INCLUDE_MODEL_OBSERVATION_NORMALIZER_HPP_

#include "running_mean_std.hpp"

namespace RLCpp {
class ObservationNormalizer;

class ObservationNormalizerImpl : public torch::nn::Module {
 private:
  torch::Tensor clip_;
  RunningMeanStd rms_;

 public:
  explicit ObservationNormalizerImpl(int size, float clip = 10.);
  ObservationNormalizerImpl(const std::vector<float> &means,
                            const std::vector<float> &variances,
                            float clip = 10.);
  explicit ObservationNormalizerImpl(const std::vector<ObservationNormalizer> &others);

  torch::Tensor process_observation(torch::Tensor observation) const;
  std::vector<float> get_mean() const;
  std::vector<float> get_variance() const;
  void update(torch::Tensor observations);

  inline float get_clip_value() const {
    return clip_.item().toFloat();
  }

  inline int get_step_count() const {
    return rms_->get_count();
  }
};
TORCH_MODULE(ObservationNormalizer);
}
#endif //EXAMPLE_APP_INCLUDE_MODEL_OBSERVATION_NORMALIZER_HPP_
