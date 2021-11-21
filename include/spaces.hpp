//
// Created by transwarp on 2021/7/7.
//

#ifndef SPACES_HPP_
#define SPACES_HPP_

#include <torch/torch.h>

namespace RLCpp {
struct ActionSpace {
  std::string type_;
  std::vector<int64_t> shape_;
};
}

#endif //EXAMPLE_APP_INCLUDE_SPACES_HPP_
