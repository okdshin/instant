cmake_minimum_required(VERSION 3.1)

# C++ setup
set(CMAKE_CXX_STANDARD 14)
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU|Clang|Intel")
	add_definitions(-g -Wall -Wextra -Wno-sign-compare -O0)
endif()
include_directories("${PROJECT_SOURCE_DIR}")

# ONNX setup
include(cmake/onnx.cmake)
include_directories(${ONNX_INCLUDE_DIR})

# MKL-DNN setup
include(cmake/mkldnn.cmake)
include_directories(${MKLDNN_INCLUDE_DIR})
link_directories(${MKLDNN_LIB_DIR})

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
add_subdirectory(instant)
