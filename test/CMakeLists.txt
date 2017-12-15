add_executable(instant_test
	test_test.cpp
	context.cpp
	onnx.cpp
	mkldnn.cpp)
target_link_libraries(instant_test
	gtest_main instant mkldnn ${PROTOBUF_LIBRARY})
add_test(NAME instant_test COMMAND instan_test)
