add_executable(instant_test
	test_test.cpp
	onnx.cpp)
target_link_libraries(instant_test
	gtest_main mkldnn onnx_proto ${PROTOBUF_LIBRARY})
add_test(NAME instant_test COMMAND instan_test)
