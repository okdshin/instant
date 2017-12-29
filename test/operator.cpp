#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <numeric>
#include <onnx.pb.h>
#include <sstream>
#include <string>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "common.hpp"
#include "np_io.hpp"

#include <instant/operator.hpp>

namespace instant {
    namespace {

        class OperatorTest : public ::testing::Test {
        protected:
            OperatorTest() = default;
            virtual void SetUp() {
                input_ = instant::load_np_array_as_array(
                  "../data/input_3_5_64_64.txt");
                engine_ = get_context().engine();
            }

            template <mkldnn::algorithm pooling_alg>
            auto pool_test_template(int k, int s, int p) const {
                auto input_memory = array_to_memory(
                  input_, mkldnn::memory::format::nchw, engine_);
                std::vector<int> stride{{s, s}};
                std::vector<int> kernel_shape{{k, k}};
                std::vector<int> padding_l{{p, p}};
                auto padding_r = padding_l;
                auto output_dims = make_conv_output_dims(
                  input_.dims(), input_.dims()[1], kernel_shape, stride,
                  padding_l, padding_r);
                auto output = array(dtype_t::float_, output_dims);
                auto output_memory = array_to_memory(
                  output, mkldnn::memory::format::nchw, engine_);
                auto net_and_temp_vars = make_pool_net<pooling_alg>(
                  input_memory, output_memory, stride, kernel_shape, padding_l,
                  padding_r, engine_);
                auto& net = std::get<0>(net_and_temp_vars);
                auto& temp_vars = std::get<1>(net_and_temp_vars);
                mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();
                auto pooling_type_str = (pooling_alg == mkldnn::pooling_max
                                           ? std::string("max_pool")
                                           : std::string("average_pool"));
                auto true_output = instant::load_np_array_as_array(
                  "../data/" + pooling_type_str + "_" + std::to_string(k) +
                  "_" + std::to_string(s) + "_" + std::to_string(p) + ".txt");
                assert_near_list(fbegin(output), fend(output),
                                 fbegin(true_output), fend(true_output),
                                 10.e-4);
            }

            array input_;
            mkldnn::engine engine_{get_context().engine()};
        };

        TEST_F(OperatorTest, max_pool_test) {
            pool_test_template<mkldnn::pooling_max>(2, 2, 0);
            pool_test_template<mkldnn::pooling_max>(3, 2, 0);
        }

        TEST_F(OperatorTest, average_pool_test) {
            pool_test_template<mkldnn::pooling_avg_include_padding>(2, 2, 0);
            pool_test_template<mkldnn::pooling_avg_include_padding>(3, 2, 0);
        }

    } // namespace
} // namespace instant
