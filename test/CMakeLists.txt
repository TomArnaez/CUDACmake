cmake_minimum_required(VERSION 3.29)

enable_testing()

find_package(GTest REQUIRED)
find_package(VulkanHeaders CONFIG)

add_executable(tests test.cpp)

target_include_directories(tests
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(tests GTest::GTest GTest::Main cudaLib Vulkan::Headers)
gtest_discover_tests(tests)