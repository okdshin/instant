include(ExternalProject)
set(EXTERNAL_DIR ${CMAKE_SOURCE_DIR}/external)
set(ONNX_DIR ${EXTERNAL_DIR}/onnx)

include(FindProtobuf)
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIR})
ExternalProject_Add(ONNX
    SOURCE_DIR ${ONNX_DIR}/src
    GIT_REPOSITORY https://github.com/onnx/onnx.git
    CONFIGURE_COMMAND ""
    BINARY_DIR ${ONNX_DIR}/build
    BUILD_COMMAND protoc -I=${ONNX_DIR}/src/onnx --cpp_out=${ONNX_DIR}/build ${ONNX_DIR}/src/onnx/onnx.proto
    INSTALL_DIR ${ONNX_DIR}/install
    INSTALL_COMMAND
        mkdir -p ${ONNX_DIR}/install/include
        COMMAND cp ${ONNX_DIR}/build/onnx.pb.h ${ONNX_DIR}/install/include/onnx.pb.h
        COMMAND cp ${ONNX_DIR}/build/onnx.pb.cc ${ONNX_DIR}/install/include/onnx.pb.cc
    LOG_CONFIGURE 1
    LOG_BUILD 1
    LOG_INSTALL 1)
set(ONNX_INCLUDE_DIR ${ONNX_DIR}/install/include CACHE PATH "Include files for ONNX")
set(PROTO_HEADER ${ONNX_DIR}/install/include/onnx.pb.h)
set(PROTO_SRC ${ONNX_DIR}/install/include/onnx.pb.cc)
add_library(onnx_proto ${PROTO_HEADER} ${PROTO_SRC})
set_source_files_properties(${PROTO_HEADER} PROPERTIES GENERATED TRUE)
set_source_files_properties(${PROTO_SRC} PROPERTIES GENERATED TRUE)
add_dependencies(onnx_proto ONNX)
