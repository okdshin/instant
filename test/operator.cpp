#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <numeric>
#include <onnx.pb.h>
#include <sstream>
#include <string>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <instant/model.hpp>

namespace instant {
    namespace {

        class OperatorTest : public ::testing::Test {};

        TEST_F(OperatorTest, vgg16_test) {
            auto onnx_model = instant::load_onnx("../data/VGG16.onnx");
            auto batch_size = 1;
            auto parameter_table = make_parameter_table(onnx_model.graph());
            auto parameter_memory_table_and_temp_array_list =
              make_parameter_memory_table(onnx_model.graph(), parameter_table,
                                          ::instant::get_context().engine());
            auto& parameter_memory_table =
              std::get<0>(parameter_memory_table_and_temp_array_list);
            auto& temp_array_list =
              std::get<1>(parameter_memory_table_and_temp_array_list);

            std::vector<
              std::tuple<std::string, instant::array, mkldnn::memory::format>>
              input_list{std::make_tuple(
                "140326425860192",
                instant::uniforms(instant::dtype_t::float_,
                                  {batch_size, 3, 224, 224}, 1.f),
                mkldnn::memory::format::nchw)};
            auto variable_memory_table = instant::make_variable_memory_table(
              input_list, ::instant::get_context().engine());

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
              onnx_model.graph(), parameter_memory_table, variable_memory_table,
              std::set<std::string>(required_output_name_list.begin(),
                                    required_output_name_list.end()));

            std::ifstream ifs("../data/vgg16_values_result.txt");
            EXPECT_TRUE(ifs);
            for(auto const& layer : required_output_name_list) {
                auto const& data = output_table[layer];

                std::string line;
                std::getline(ifs, line);
                std::istringstream iss(line);
                std::vector<float> true_values;
                std::string layer_name;
                iss >> layer_name;
                for(int j = 0; j < instant::total_size(data); ++j) {
                    float v;
                    iss >> v;
                    true_values.push_back(v);
                }

                for(int i = 0; i < instant::total_size(data); ++i) {
                    auto x = *(static_cast<float const*>(data.data()) + i);
                    auto t = true_values[i];
                    ASSERT_NEAR(t, x, 10e-4) << layer << " " << i;
                }
            }
        }

        TEST_F(OperatorTest, resnet50_test) {
            auto onnx_model = instant::load_onnx("../data/ResNet50.onnx");
            auto batch_size = 1;
            auto parameter_table = make_parameter_table(onnx_model.graph());
            auto parameter_memory_table_and_temp_array_list =
              make_parameter_memory_table(onnx_model.graph(), parameter_table,
                                          ::instant::get_context().engine());
            auto& parameter_memory_table =
              std::get<0>(parameter_memory_table_and_temp_array_list);
            auto& temp_array_list =
              std::get<1>(parameter_memory_table_and_temp_array_list);
            std::cout << "parameter memory table" << std::endl;
            for(auto const& p : parameter_memory_table) {
                std::cout << p.first << std::endl;
            }

            std::vector<
              std::tuple<std::string, instant::array, mkldnn::memory::format>>
              input_list{std::make_tuple(
                "140555732057560",
                instant::uniforms(instant::dtype_t::float_,
                                  {batch_size, 3, 224, 224}, 1.f),
                mkldnn::memory::format::nchw)};
            auto variable_memory_table = instant::make_variable_memory_table(
              input_list, ::instant::get_context().engine());

            std::vector<std::string> required_output_name_list{
              "140555506372504", // conv1
              "140555734620144", // pool1
            };
            auto output_table = run_model(
              onnx_model.graph(), parameter_memory_table, variable_memory_table,
              std::set<std::string>(required_output_name_list.begin(),
                                    required_output_name_list.end()));

            std::ifstream ifs("../data/resnet50_values_result.txt");
            EXPECT_TRUE(ifs);
            for(auto const& layer : required_output_name_list) {
                auto const& data = output_table[layer];

                std::string line;
                std::getline(ifs, line);
                std::istringstream iss(line);
                std::vector<float> true_values;
                std::string layer_name;
                iss >> layer_name;
                for(int j = 0; j < instant::total_size(data); ++j) {
                    float v;
                    iss >> v;
                    true_values.push_back(v);
                }

                ASSERT_EQ(instant::total_size(data), true_values.size());
                std::cout << *(static_cast<float const*>(data.data()) + 56) << std::endl;
                for(int i = 0; i < instant::total_size(data); ++i) {
                    auto x = *(static_cast<float const*>(data.data()) + i);
                    auto t = true_values.at(i);
                   // std::cout << x << " " << t << std::endl;
                    ASSERT_NEAR(t, x, 10e-4) << layer << " " << i;
                }
                std::cout << "here" << std::endl;
            }
        }

        /*
        TEST_F(OperatorTest, max_pool_test) {
            auto batch_size = 1;
            std::vector<
              std::tuple<std::string, instant::array, mkldnn::memory::format>>
              input_list{std::make_tuple(
                "140326425860192",
                instant::uniforms(instant::dtype_t::float_,
                                  {batch_size, 3, 224, 224}, 1.f),
                mkldnn::memory::format::nchw)};
            auto variable_memory_table = instant::make_variable_memory_table(
              input_list, ::instant::get_context().engine());

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
              onnx_model.graph(), parameter_memory_table, variable_memory_table,
              std::set<std::string>(required_output_name_list.begin(),
                                    required_output_name_list.end()));

            std::ifstream ifs("../data/vgg16_values_result.txt");
            EXPECT_TRUE(ifs);
            for(auto const& layer : required_output_name_list) {
                auto const& data = output_table[layer];

                std::string line;
                std::getline(ifs, line);
                std::istringstream iss(line);
                std::vector<float> true_values;
                std::string layer_name;
                iss >> layer_name;
                for(int j = 0; j < instant::total_size(data); ++j) {
                    float v;
                    iss >> v;
                    true_values.push_back(v);
                }

                for(int i = 0; i < instant::total_size(data); ++i) {
                    auto x = *(static_cast<float const*>(data.data()) + i);
                    auto t = true_values[i];
                    ASSERT_NEAR(t, x, 10e-4) << layer << " " << i;
                }
            }
        }
        */

    } // namespace
} // namespace instant
