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
    // auto onnx_model = instant::load_onnx("../data/VGG.onnx");
    auto batch_size = 1;
    auto onnx_model = instant::load_onnx("../data/vgg16/model.pb");
    auto parameter_table = make_parameter_table(onnx_model.graph());
    std::unordered_map<std::string, instant::array> input_table;
    input_table["gpu_0/data_0"] = instant::uniforms(
        instant::dtype_t::float_, {batch_size, 3, 224, 224}, 0);
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
    auto output_table =
        run_model(onnx_model.graph(), parameter_memory_table,
                  variable_memory_table, {"gpu_0/conv1_1", "gpu_0/conv1_2"});
    for (auto const& p : output_table) {
        std::cout << p.first << std::endl;
        for (auto d : p.second.dims()) {
            std::cout << d << " ";
        }
        std::cout << std::endl;
        //for (int i = 0; i < instant::calc_total_size(p.second.dims()); ++i) {
        for (int i = 0; i < 10; ++i) {
            std::cout << *(static_cast<float const*>(p.second.data()) + i)
                      << " ";
        }
        std::cout << std::endl;
    }
}

}  // namespace
}  // namespace instant
