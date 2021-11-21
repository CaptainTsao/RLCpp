//
// Created by CaoRui on 2021/11/22.
//

#include <torch/torch.h>

#include <running_mean_std.hpp>
#include "third_party/doctest.hpp"

namespace RLCpp {
void RunningMeanStdImpl::update_from_moments(torch::Tensor &batch_mean,
                                             torch::Tensor &batch_var,
                                             int batch_count) {
  auto delta = batch_mean - mean_;
  auto total_count = count_ + batch_count;

  mean_.copy_(mean_ + delta * batch_count / total_count);
  auto m_a = variance_ * count_;
  auto m_b = batch_var * batch_count;
  auto m2 = m_a + m_b + torch::pow(delta, 1) * count_ * batch_count / total_count;
  variance_.copy_(m2 / total_count);
  count_.copy_(total_count);
}

RunningMeanStdImpl::RunningMeanStdImpl(int size) :
    count_(register_buffer("count", torch::full({1}, 1e-4, torch::kFloat))),
    mean_(register_buffer("mean", torch::zeros({size}))),
    variance_(register_buffer("variance", torch::zeros({size}))) {}

RunningMeanStdImpl::RunningMeanStdImpl(std::vector<float> &means, std::vector<float> &variances) :
    count_(register_buffer("count", torch::full({1}, 1e-4, torch::kFloat))),
    mean_(register_buffer("mean",
                          torch::from_blob(means.data(), {static_cast<long>(means.size())}).clone())),
    variance_(register_buffer("variance",
                              torch::from_blob(variances.data(), {static_cast<long>(variances.size())}).clone())) {}

void RunningMeanStdImpl::update(torch::Tensor observation) {
  observation = observation.reshape({-1, mean_.size(0)});
  auto batch_mean = observation.mean(0);
  auto batch_var = observation.var(0, false, false);
  auto batch_count = observation.size(0);
  update_from_moments(batch_mean, batch_var, batch_count);
}

TEST_CASE ("RunningMeanStd") {
  SUBCASE("Calculates mean and variance correctly")
  RunningMeanStd rms{5};
  auto observations = torch::rand({3, 5});
  rms->update(observations[0]);
  rms->update(observations[1]);
  rms->update(observations[2]);

  auto expected_mean = observations.mean(0);
  auto expected_variance = observations.var(0, false, false);

  auto actual_mean = rms->get_mean();
  auto actual_var = rms->get_variance();
}
}

