target_sources(RLCpp
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/actorcritic.cpp
        ${CMAKE_CURRENT_LIST_DIR}/proximal_policy_optimization.cpp
        )

if (CPPRL_BUILD_TESTS)
    target_sources(rlcpp_tests
            PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/actorcritic.cpp
            ${CMAKE_CURRENT_LIST_DIR}/proximal_policy_optimization.cpp
            )
endif (CPPRL_BUILD_TESTS)