target_sources(RLCpp
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/storage.cpp
        ${CMAKE_CURRENT_LIST_DIR}/observation_normalizer.cpp
        ${CMAKE_CURRENT_LIST_DIR}/running_mean_std.cpp
        )

if(RLCPP_BUILD_TESTS)
    target_sources(rlcpp_tests
            PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/storage.cpp
            ${CMAKE_CURRENT_LIST_DIR}/observation_normalizer.cpp
            ${CMAKE_CURRENT_LIST_DIR}/running_mean_std.cpp
            )
endif()

add_subdirectory(algorithm)
add_subdirectory(distribution)
add_subdirectory(generators)
add_subdirectory(model)
add_subdirectory(third_party)