cmake_minimum_required(VERSION 3.1)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# C++ setup
set(CMAKE_CXX_STANDARD 17)
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU|Clang|Intel")
	add_definitions(-Wall -Wextra -Wno-sign-compare -O3)
endif()
include_directories("${PROJECT_SOURCE_DIR}")

# ONNX setup
include(cmake/onnx.cmake)
include_directories("${PROJECT_SOURCE_DIR}/instant")

# MKLDNN setup
find_package(MKLDNN)
if (NOT MKLDNN_FOUND)
    message(FATAL_ERROR "MKLDNN is not found")
endif()

include_directories("${MKLDNN_INCLUDE_DIR}")

if(${DISABLE_TEST})
    set(ENABLE_TEST OFF)
else()
    set(ENABLE_TEST ON)
endif()
if(${ENABLE_TEST})
    include(cmake/gtest.cmake)
    enable_testing()
    add_subdirectory(test)
endif()

add_subdirectory(example)
add_subdirectory(tool)
add_subdirectory(instant)

