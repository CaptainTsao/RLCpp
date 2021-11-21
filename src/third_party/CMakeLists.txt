if (CPPRL_BUILD_TESTS)
    target_sources(rlcpp_tests
            PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/doctest.cpp
            )
endif (CPPRL_BUILD_TESTS)