include(ExternalProject)
set(EXTERNAL_DIR ${CMAKE_SOURCE_DIR}/external)
set(ONNX_DIR ${EXTERNAL_DIR}/onnx)
set(Protobuf_PROTOC_EXECUTABLE protoc CACHE STRING "protoc path")

include(FindProtobuf)
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIR})
ExternalProject_Add(
    ONNX
    SOURCE_DIR ${ONNX_DIR}/src
    GIT_REPOSITORY https://github.com/onnx/onnx.git
    CONFIGURE_COMMAND ""
    BINARY_DIR ${ONNX_DIR}/build
    BUILD_COMMAND ${Protobuf_PROTOC_EXECUTABLE} -I=${ONNX_DIR}/src/onnx --cpp_out=${ONNX_DIR}/build ${ONNX_DIR}/src/onnx/onnx.proto
    INSTALL_DIR ${ONNX_DIR}/install
    INSTALL_COMMAND cp ${ONNX_DIR}/build/onnx.pb.h ${CMAKE_SOURCE_DIR}/instant/onnx.pb.h
            COMMAND cp ${ONNX_DIR}/build/onnx.pb.cc ${CMAKE_SOURCE_DIR}/instant/onnx.pb.cc
    LOG_CONFIGURE 1
    LOG_BUILD 1
    LOG_INSTALL 1)
