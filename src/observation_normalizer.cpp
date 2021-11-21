//
// Created by CaoRui on 2021/11/22.
//

#include <torch/torch.h>

#include "observation_normalizer.hpp"
#include "running_mean_std.hpp"
#include "third_party/doctest.hpp"

namespace RLCpp {

ObservationNormalizerImpl::ObservationNormalizerImpl(int size, float clip)
    : clip_(register_buffer("clip", torch::full({1}, clip, torch::kFloat))),
      rms_(register_module("rms", RunningMeanStd(size))) {}

ObservationNormalizerImpl::ObservationNormalizerImpl(const std::vector<float> &means,
                                                     const std::vector<float> &variances,
                                                     float clip) :
    clip_(register_buffer("clip", torch::full({1}, clip, torch::kFloat))),
    rms_(register_module("rms", RunningMeanStd(means, variances))) {}

ObservationNormalizerImpl::ObservationNormalizerImpl(const std::vector<ObservationNormalizer> &others) :
    clip_(register_buffer("clip", torch::zeros({1}, torch::kFloat))),
    rms_(register_module("rms", RunningMeanStd(1))) {
  // Calculate mean clip
  for (const auto &other:others) {
    clip_ += other->get_clip_value();
  }
  clip_[0] = clip_[0] / static_cast<float>(others.size());
  // Calculate mean mean
  std::vector<float> mean_means(others[0]->get_mean().size(), 0);
  for (const auto &other:others) {
    auto other_mean = other->get_mean();
    for (unsigned int i = 0; i < mean_means.size(); ++i) {
      mean_means[i] += other_mean[i];
    }
  }
  for (auto &mean:mean_means) {
    mean /= static_cast<float>(others.size());
  }

  // calculate mean variances
  std::vector<float> mean_variances(others[0]->get_variance().size(), 0);
  for (const auto &other:others) {
    auto other_variances = other->get_variance();
    for (unsigned int i = 0; i < mean_variances.size(); ++i) {
      mean_variances[i] += other_variances[i];
    }
  }
  for (auto &variance : mean_variances) {
    variance /= static_cast<float>(others.size());
  }

  rms_ = RunningMeanStd(mean_means, mean_variances);
  int total_count = std::accumulate(
      others.begin(), others.end(), 0,
      [](int accumulator, const ObservationNormalizer &other) {
        return accumulator + other->get_step_count();
      }
  );
  rms_->set_count(total_count);
}

torch::Tensor ObservationNormalizerImpl::process_observation(torch::Tensor &observation) const {
  auto normalized_obs = (observation - rms_->get_mean()) / torch::sqrt(rms_->get_variance() + 1e-8);
  return torch::clamp(normalized_obs, -clip_.item(), clip_.item());
}

std::vector<float> ObservationNormalizerImpl::get_mean() const {
  auto mean = rms_->get_mean();
  return std::vector<float>(mean.data_ptr<float>(), mean.data_ptr<float>() + mean.numel());
}

std::vector<float> ObservationNormalizerImpl::get_variance() const {
  auto variance = rms_->get_variance();
  return std::vector<float>(variance.data_ptr<float>(), variance.data_ptr<float>() + variance.numel());
}

void ObservationNormalizerImpl::update(torch::Tensor &observations) {
  rms_->update(observations);
}

TEST_CASE ("ObservationNormalizer") {
  SUBCASE("Clips values correctly")
  {
    ObservationNormalizer normalizer(7, 1);
    float observation_float[] = {-1000, -100, -10, 0, 10, 100, 1000};
    auto observation = torch::from_blob(observation_float, {7});
    auto processed_observation = normalizer->process_observation(observation);

    auto has_too_large_values = (processed_observation > 1).any().item().toBool();
    auto has_too_small_values = (processed_observation < -1).any().item().toBool();
    DOCTEST_CHECK(!has_too_large_values);
    DOCTEST_CHECK(!has_too_small_values);
  }
}
}