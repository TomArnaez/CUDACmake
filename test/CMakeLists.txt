cmake_minimum_required(VERSION 3.29)

enable_testing()

find_package(GTest REQUIRED)
find_package(OpenCV REQUIRED)
find_package(VulkanHeaders CONFIG)

add_executable(tests test.cpp)

set_target_properties(tests PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

target_include_directories(tests
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    tests 
        PRIVATE   
            cudaLib GTest::GTest GTest::Main Vulkan::Headers ${OpenCV_LIBS}
)

gtest_discover_tests(tests)