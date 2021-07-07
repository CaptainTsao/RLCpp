cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(example_app CXX C)

set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch")

set(CMAKE_CXX_STANDARD 14)
#message(${CMAKE_CURRENT_SOURCE_DIR})
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}${TORCH_CXX_FLAGS}")

list(APPEND LIGHTSTEP_LINK_LIBRARY glog pthread )

add_executable(example_app
        main.cpp
        include/spaces.hpp
        include/generators/generator.hpp include/storage.hpp storage.cpp include/algorithm/algorithm.hpp include/algorithm/actorcritic.hpp include/model/model_utils.hpp include/model/nn_base.hpp include/model/mlp_base.hpp src/model/model_utils.cpp)
#autoencoder.cpp autoencoder.h VAE_test.cpp
#add_executable(example_app example_app.cpp)
target_link_libraries(example_app "${TORCH_LIBRARIES}")
set_property(TARGET example_app PROPERTY CXX_STANDARD 14)

