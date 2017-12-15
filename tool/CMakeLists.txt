add_executable(onnx_viewer onnx_viewer.cpp)
target_link_libraries(onnx_viewer instant ${PROTOBUF_LIBRARY})
