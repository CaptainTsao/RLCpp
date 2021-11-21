//
// Created by transwarp on 2021/7/8.
//

#ifndef RUNNING_MEAN_STD_HPP_
#define RUNNING_MEAN_STD_HPP_

#include <torch/torch.h>

namespace RLCpp {
class RunningMeanStdImpl : public torch::nn::Module {
 private:
  torch::Tensor count_, mean_, variance_;
  void update_from_moments(torch::Tensor &batch_mean,
                           torch::Tensor &batch_var,
                           int batch_count);

 public:
  explicit RunningMeanStdImpl(int size);
  RunningMeanStdImpl(std::vector<float> &means, std::vector<float> &variances);

  void update(torch::Tensor observation);

  inline int get_count() const {
    return static_cast<int>(count_.item().toFloat());
  }
  inline torch::Tensor get_mean() const {
    return mean_.clone();
  }
  inline torch::Tensor get_variance() const {
    return variance_.clone();
  }
  inline void set_count(int count) {
    this->count_[0] = count + 1e-8;
  }
};
TORCH_MODULE(RunningMeanStd);
}

#endif //EXAMPLE_APP_INCLUDE_RUNNING_MEAN_STD_HPP_
