file(GLOB src "*.cc" "*.cpp")
add_library(instant ${src})
add_dependencies(instant ONNX)

install(DIRECTORY ./ DESTINATION include/instant FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
install(TARGETS instant DESTINATION lib)
