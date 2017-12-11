#include <gtest/gtest.h>
#include <onnx.pb.h>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <instant/model.hpp>

namespace instant {
namespace {

class MKLDNNTest : public ::testing::Test {};

TEST_F(MKLDNNTest, run_onnx_model) {
    //auto onnx_model = instant::load_onnx("../data/VGG.onnx");
    auto batch_size = 1;
    auto onnx_model = instant::load_onnx("../data/vgg16/model.pb");
    auto parameter_table = make_parameter_table(onnx_model.graph());
    std::unordered_map<std::string, instant::array> input_table;
    input_table["gpu_0/data_0"] = instant::array(
        instant::dtype_t::float_, {batch_size, 3, 224, 224},
        std::unique_ptr<float[]>(new float[batch_size * 3 * 224 * 224]));
    /*
    std::unordered_map<std::string, instant::array> output_table;
    output_table["gpu_0/pred_1"] =
        instant::array(instant::dtype_t::float_, {batch_size, 1000},
                       std::unique_ptr<float[]>(new float[batch_size * 1000]));
    */
    /*
    std::map<std::string, instant::array> initializers =
    load_initializers(onnx_model.graph()); std::vector<onnx::NodeProto> nodes =
    load_nodes(onnx_model.graph()); std::vector<std::vector<onnx::NodeProto
    const*>> node_partial_order = serialize_nodes(initializers, nodes);
    instant::construct_net(initializers, node_partial_order);
    */
    // instant::construct_net(onnx_model.graph(), parameter_table,
    // variable_table);
    auto parameter_memory_table = instant::make_parameter_memory_table(
        onnx_model.graph(), parameter_table, ::instant::get_context().engine());
    std::cout << "parameter memory table" << std::endl;
    for (auto const& p : parameter_memory_table) {
        std::cout << p.first << std::endl;
    }

    std::vector<std::tuple<std::string, instant::array, mkldnn::memory::format>>
        input_list{{"gpu_0/data_0", input_table["gpu_0/data_0"],
                    mkldnn::memory::format::nchw}};
    auto variable_memory_table = instant::make_variable_memory_table(
        input_list, ::instant::get_context().engine());
    std::cout << "variable memory table" << std::endl;
    for (auto const& p : variable_memory_table) {
        std::cout << p.first << std::endl;
    }
    std::cout << "array " << *static_cast<float*>(parameter_table["gpu_0/conv1_w_0"].data()) << std::endl;
    std::cout << "array " << static_cast<float*>(variable_memory_table.find("gpu_0/data_0")->second.get_data_handle()) << std::endl;
    auto output_table = run_model(onnx_model.graph(), parameter_memory_table,
                                  variable_memory_table, {"gpu_0/conv1_1"});
    std::cout << "array " << *static_cast<float*>(parameter_table["gpu_0/conv1_w_0"].data()) << std::endl;
    for (auto const& p : output_table) {
        std::cout << p.first << std::endl;
    }
}

}  // namespace
}  // namespace instant
