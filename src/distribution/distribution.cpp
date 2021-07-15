//
// Created by transwarp on 2021/7/8.
//

#include "../../include/distributions/distribution.hpp"

namespace RLCpp {
std::vector<int64_t> Distribution::extended_shape(const c10::ArrayRef<int64_t> &sample_shape) {
  std::vector<int64_t> output_shape;
  output_shape.insert(output_shape.end(),
                      sample_shape.begin(),
                      sample_shape.end());
  output_shape.insert(output_shape.end(),
                      batch_shape_.begin(),
                      batch_shape_.end());
  output_shape.insert(output_shape.end(),
                      batch_shape_.begin(),
                      batch_shape_.end());
  return output_shape;
}
}


