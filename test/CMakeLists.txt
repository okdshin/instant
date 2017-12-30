add_executable(instant_test
    np_io.cpp
    context.cpp
    onnx.cpp
    mkldnn.cpp
    operator.cpp
    node.cpp
)
target_link_libraries(instant_test
    gtest_main instant ${MKLDNN_LIBRARY} ${PROTOBUF_LIBRARY})
add_test(NAME instant_test COMMAND instan_test)
