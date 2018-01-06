#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <numeric>
#include <onnx.pb.h>
#include <string>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <instant/mkldnn/model.hpp>
#include <instant/onnx.hpp>

namespace instant {
    namespace {

        class MKLDNNTest : public ::testing::Test {};

        /*
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
        */

        TEST_F(MKLDNNTest, run_onnx_model) {
            std::set<std::string> required_output_name_set{"140326200803680"};
            auto[graph, param_table, input_name_set] = load_onnx(
              "../data/VGG16.onnx", required_output_name_set);
            if(input_name_set.size() != 1) {
                throw std::runtime_error("VGG16 data is invalid");
            }
            auto const& input_name = *input_name_set.begin();
            constexpr auto batch_size = 1;
            constexpr auto channel_num = 3;
            constexpr auto height = 224;
            constexpr auto width = 224;
            std::vector<int> input_dims{batch_size, channel_num, height, width};
            array input_arr(dtype_t::float_, input_dims);
            auto nets = mkldnn_backend::make_nets(
              graph, param_table, {{input_name, input_arr},}, required_output_name_set);
        }

    } // namespace
} // namespace instant
