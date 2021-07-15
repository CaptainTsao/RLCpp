//
// Created by transwarp on 2021/7/7.
//

#ifndef EXAMPLE_APP_INCLUDE_ALGORITHM_ALGORITHM_HPP_
#define EXAMPLE_APP_INCLUDE_ALGORITHM_ALGORITHM_HPP_

#include "../storage.hpp"

namespace RLCpp {
struct UpdateDatum {
  std::string name_;
  float value_;
};

class Algorithm {
 public:
  virtual ~Algorithm() = 0;
  virtual std::vector<UpdateDatum> Update(RolloutStorage &rollout_storage,
                                          float decay_level) = 0;
};
inline Algorithm::~Algorithm() = default;
}

#endif //EXAMPLE_APP_INCLUDE_ALGORITHM_ALGORITHM_HPP_
