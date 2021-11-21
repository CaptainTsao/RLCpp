cmake_minimum_required(VERSION 3.15.0)
project(RLCpp
        LANGUAGES CXX C
        VERSION 1.0.0
        DESCRIPTION "Reinforcement learning in C++ using PyTorch"
        )

# Project-wide properties
set(CMAKE_CXX_STANDARD 14)

# Cppcheck
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    list(APPEND CPPCHECK_ARGS
            --enable=warning
            --std=c++14
            --force
            --verbose
            --quiet
            --inline-suppr
            --error-exitcode=1
            --language=c++
            --config-exclude=${CMAKE_CURRENT_LIST_DIR}/src/third_party
            --config-exclude=${CMAKE_CURRENT_LIST_DIR}/lib
            -i${CMAKE_CURRENT_LIST_DIR}/example/lib
            --suppressions-list=${CMAKE_CURRENT_LIST_DIR}/CppCheckSuppressions.txt
            -I ${CMAKE_CURRENT_LIST_DIR}/src
            -I ${CMAKE_CURRENT_LIST_DIR}/include
            -I ${CMAKE_CURRENT_LIST_DIR}/example
            ${CMAKE_CURRENT_LIST_DIR}/src
            ${CMAKE_CURRENT_LIST_DIR}/example
            )

    add_custom_target(
            check
            COMMAND cppcheck ${CPPCHECK_ARGS}
            COMMENT "Running Cppcheck"
    )
endif (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)

# Dependencies
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch")
## PyTorch
if (NOT TORCH_FOUND)
    find_package(Torch REQUIRED)
    if (TORCH_CXX_FLAGS)
        set(CMAKE_CXX_FLAGS ${TORCH_CXX_FLAGS})
    endif ()
endif (NOT TORCH_FOUND)

# Define targets
add_library(RLCpp STATIC "")
target_compile_definitions(RLCpp PRIVATE DOCTEST_CONFIG_DISABLE)
option(CPPRL_BUILD_TESTS "Whether or not to build the RLCpp tests" OFF)
if (CPPRL_BUILD_TESTS)
    add_executable(rlcpp_tests "")
endif (CPPRL_BUILD_TESTS)

# Enable all warnings
if (MSVC)
    target_compile_options(RLCpp PRIVATE /W0)
else (MSVC)
    target_compile_options(RLCpp PRIVATE -Wall -Wextra -pedantic)
endif (MSVC)

# Includes
set(CPPRL_INCLUDE_DIRS
    include
    src
    ${TORCH_INCLUDE_DIRS}
    )
target_include_directories(RLCpp PRIVATE ${CPPRL_INCLUDE_DIRS})
if (CPPRL_BUILD_TESTS)
    target_include_directories(rlcpp_tests PRIVATE ${CPPRL_INCLUDE_DIRS})
endif (CPPRL_BUILD_TESTS)

# Linking
target_link_libraries(RLCpp torch ${TORCH_LIBRARIES})
target_link_libraries(RLCpp torch ${TORCH_LIBRARIES})
if (CPPRL_BUILD_TESTS)
    target_link_libraries(rlcpp_tests torch ${TORCH_LIBRARIES})
endif (CPPRL_BUILD_TESTS)

# Example
option(CPPRL_BUILD_EXAMPLE "Whether or not to build the RLCpp Gym example" ON)
if (CPPRL_BUILD_EXAMPLE)
    add_subdirectory(example)
endif (CPPRL_BUILD_EXAMPLE)

# Recurse into source tree
add_subdirectory(src)
