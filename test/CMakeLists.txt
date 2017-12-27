add_executable(instant_test
    test_test.cpp
    context.cpp
    onnx.cpp
    mkldnn.cpp
    operator.cpp
)
target_link_libraries(instant_test
    gtest_main instant ${MKLDNN_LIBRARY} ${PROTOBUF_LIBRARY})
add_test(NAME instant_test COMMAND instan_test)

add_executable(operator_test
    operator.cpp
)
target_link_libraries(operator_test
    gtest_main instant ${MKLDNN_LIBRARY} ${PROTOBUF_LIBRARY})
add_test(NAME operator_test COMMAND operator_test)
