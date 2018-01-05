#include <gtest/gtest.h>
#include <onnx.pb.h>
#include <fstream>
#include <instant/onnx.hpp>
#include <iostream>
#include <numeric>
#include <string>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

namespace instant {
namespace {

class ONNXTest : public ::testing::Test {};

TEST_F(ONNXTest, load_onnx_model) {
    auto [graph, param_table] = instant::load_onnx("../data/VGG16.onnx", {"140326201105432", /*"140326200803456"*//*, "140326200803680"*/});
    std::cout << "param table" << std::endl;
    for(auto [name, arr] : param_table) {
        std::cout << name << " ";
       for(auto d : arr.dims()) {
           std::cout << d << " ";
       }
       std::cout << "\n";
    }
    std::cout << "graph" << std::endl;
    for(auto const& node_set : graph) {
        for(auto const& node : node_set) {
            std::cout << op_type_to_string(node.op_type()) << " ";
        }
        std::cout << std::endl;
    }
}

}  // namespace
}  // namespace instant
