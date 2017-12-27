#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <numeric>
#include <onnx.pb.h>
#include <string>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <instant/model.hpp>

namespace instant {
    namespace {

        class MKLDNNTest : public ::testing::Test {};

        TEST_F(MKLDNNTest, test_calc_reshaped_dims) {
            {
                auto new_shape = instant::calc_reshaped_dims({10, 20, 30, 40},
                                                             {10, 20, -1, 40});
                for(auto n : new_shape) {
                    std::cout << n << " ";
                }
                std::cout << "\n";
            }
            {
                auto new_shape =
                  instant::calc_reshaped_dims({10, 20, 30, 40}, {10, -1});
                for(auto n : new_shape) {
                    std::cout << n << " ";
                }
                std::cout << "\n";
            }
            {
                auto new_shape = instant::calc_reshaped_dims({10, 20, 30, 40},
                                                             {40, 20, 30, 10});
                for(auto n : new_shape) {
                    std::cout << n << " ";
                }
                std::cout << "\n";
            }
        }

        TEST_F(MKLDNNTest, run_onnx_model) {
            auto onnx_model = instant::load_onnx("../data/VGG16.onnx");
            auto batch_size = 1;
            // auto onnx_model = instant::load_onnx("../data/vgg16/model.pb");
            auto parameter_table = make_parameter_table(onnx_model.graph());
            for(auto const& p : parameter_table) {
                std::cout << p.first << std::endl;
            }
            auto parameter_memory_table_and_temp_array_list =
              instant::make_parameter_memory_table(
                onnx_model.graph(), parameter_table,
                ::instant::get_context().engine());
            auto parameter_memory_table =
              std::get<0>(parameter_memory_table_and_temp_array_list);
            auto temp_array_list =
              std::get<1>(parameter_memory_table_and_temp_array_list);

            std::cout << "parameter memory table" << std::endl;
            for(auto const& p : parameter_memory_table) {
                std::cout << p.first << std::endl;
            }

            std::vector<
              std::tuple<std::string, instant::array, mkldnn::memory::format>>
              input_list{
                std::make_tuple("140326425860192",
                                instant::uniforms(instant::dtype_t::float_,
                                                  {batch_size, 3, 224, 224}, 1),
                                mkldnn::memory::format::nchw)};
            auto variable_memory_table = instant::make_variable_memory_table(
              input_list, ::instant::get_context().engine());
            std::cout << "variable memory table" << std::endl;
            for(auto const& p : variable_memory_table) {
                std::cout << p.first << std::endl;
            }
            std::vector<std::string> required_output_name_list{
              "140326201105432", // conv1_1
              "140326201105600", // conv1_2
              "140326429223512", // pool1
              "140326150903400", // conv2_1
              "140326200661440", // conv2_2
              "140326200661720", // pool2
              "140326200662112", // conv3_1
              "140326200662560", // conv3_2
              "140326200663008", // conv3_3
              "140326200663288", // pool3
              "140326200663680", // conv4_1
              "140326200774784", // conv4_2
              "140326200775232", // conv4_3
              "140326200775512", // pool4
              "140326200775904", // conv5_1
              "140326200776352", // conv5_2
              "140326200776800", // conv5_3
              "140326200777080", // pool5
              "140326200777976", // fc6
              "140326200778648", // fc7
              "140326200803456", // fc8
              "140326200803680", // prob
            };
            auto output_table = run_model(
              onnx_model.graph(), parameter_memory_table,
              // variable_memory_table, {"gpu_0/conv1_1", "gpu_0/conv1_2"});
              variable_memory_table,
              std::set<std::string>(required_output_name_list.begin(),
                                    required_output_name_list.end()));
            for(auto const& layer : required_output_name_list) {
                auto const& data = output_table[layer];
                for(int i = 0; i < 10; ++i) {
                    std::cout << *(static_cast<float const*>(data.data()) + i)
                              << " ";
                }
                std::cout << "\n";
            }
            /*
            for (auto const& p : output_table) {
                std::cout << p.first << " (";
                for (auto d : p.second.dims()) {
                    std::cout << d << " ";
                }
                std::cout << ")" << std::endl;
                // for (int i = 0; i <
            instant::calc_total_size(p.second.dims()); ++i) { for (int i = 0; i
            < 10; ++i) { std::cout << *(static_cast<float
            const*>(p.second.data()) + i)
                              << " ";
                }
                std::cout << std::endl;
            }
            */
        }

    } // namespace
} // namespace instant
