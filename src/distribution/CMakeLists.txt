target_sources(RLCpp
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/bernoulli.cpp
        ${CMAKE_CURRENT_LIST_DIR}/categorical.cpp
        ${CMAKE_CURRENT_LIST_DIR}/distribution.cpp
        ${CMAKE_CURRENT_LIST_DIR}/normal.cpp
        )

if (CPPRL_BUILD_TESTS)
    target_sources(rlcpp_tests
            PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/bernoulli.cpp
            ${CMAKE_CURRENT_LIST_DIR}/categorical.cpp
            ${CMAKE_CURRENT_LIST_DIR}/distribution.cpp
            ${CMAKE_CURRENT_LIST_DIR}/normal.cpp
            )
endif (CPPRL_BUILD_TESTS)