file(GLOB src "onnx.pb.cc" "*.cpp")
set_source_files_properties(${CMAKE_SOURCE_DIR}/instant/onnx.pb.cc PROPERTIES GENERATED TRUE)
add_library(instant "${CMAKE_SOURCE_DIR}/instant/onnx.pb.cc" "${src}")
add_dependencies(instant ONNX)

install(DIRECTORY ./ DESTINATION include/instant FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
install(TARGETS instant DESTINATION lib)
