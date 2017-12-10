#include <gtest/gtest.h>
#include <onnx.pb.h>
#include <fstream>
#include <instant/load_onnx.hpp>
#include <iostream>
#include <numeric>
#include <string>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

namespace instant {
namespace {

class ONNXTest : public ::testing::Test {};

#ifdef AAAA
TEST_F(ONNXTest, load_onnx_model_and_parse) {
    namespace gpio = ::google::protobuf::io;

    std::ifstream ifs("../data/VGG.onnx");
    gpio::IstreamInputStream iis(&ifs);
    gpio::CodedInputStream cis(&iis);
    cis.SetTotalBytesLimit(std::numeric_limits<int>::max(),
                           std::numeric_limits<int>::max());

    onnx::ModelProto model;
    if (!model.ParseFromCodedStream(&cis)) {
        EXPECT_TRUE(false) << "model load error";
    }
    std::cerr << model.has_ir_version() << std::endl;
    std::cerr << model.ir_version() << std::endl;
    std::cerr << model.has_producer_name() << std::endl;
    std::cerr << model.producer_name() << std::endl;
    std::cerr << model.doc_string() << std::endl;
    std::cerr << model.has_graph() << std::endl;

    auto graph = model.graph();
    std::cerr << graph.node_size() << std::endl;
    for (auto const& tensor : graph.initializer()) {
        std::cerr << tensor.name() << std::endl;
        for (int j = 0; j < tensor.dims_size(); ++j) {
            std::cerr << tensor.dims(j) << " ";
        }
        std::cerr << "\n";
    }
    for (int i = 0; i < graph.node_size(); ++i) {
        auto node = graph.node(i);
        std::cerr << i << ":" << node.op_type() << std::endl;
        for (int j = 0; j < node.input_size(); ++j) {
            std::cerr << "\ti" << j << ":" << node.input(j) << std::endl;
            std::cerr << "\ti" << j << ":" << *node.mutable_input(j)
                      << std::endl;
        }
        for (int j = 0; j < node.output_size(); ++j) {
            std::cerr << "\to" << j << ":" << node.output(j) << std::endl;
        }
        for (int j = 0; j < node.attribute_size(); ++j) {
            auto attribute = node.attribute(j);
            std::cerr << "\ta" << j << ":" << attribute.name()
                      << AttributeProto_AttributeType_Name(attribute.type())
                      << std::endl;
            if (attribute.has_f()) {
                std::cerr << "\t\tf"
                          << ":" << attribute.f() << std::endl;
            }
            if (attribute.has_i()) {
                std::cerr << "\t\ti"
                          << ":" << attribute.i() << std::endl;
            }
            if (attribute.has_s()) {
                std::cerr << "\t\ts"
                          << ":" << attribute.s() << std::endl;
            }
            if (attribute.floats_size()) {
                for (int k = 0; k < attribute.floats_size(); ++k) {
                    std::cerr << "\t\tfs" << attribute.floats(k) << " ";
                }
                std::cerr << "\n";
            }
            if (attribute.ints_size()) {
                for (int k = 0; k < attribute.ints_size(); ++k) {
                    std::cerr << "\t\tis" << attribute.ints(k) << " ";
                }
                std::cerr << "\n";
            }
            if (attribute.strings_size()) {
                for (int k = 0; k < attribute.strings_size(); ++k) {
                    std::cerr << "\t\tis" << attribute.strings(k) << " ";
                }
                std::cerr << "\n";
            }
            if (attribute.tensors_size()) {
                std::cerr << "\t\ttensors";
            }
            /*
            if (attribute.has_t()) {
                std::cerr << "\t\tt" << j << ":" << attribute.t() << std::endl;
            }
            */
        }
    }
}
#endif

/*
TEST_F(ONNXTest, load_onnx_model) {
    std::map<std::string, instant::array> initializers;
    std::vector<onnx::NodeProto> nodes;
    std::vector<std::vector<onnx::NodeProto const*>> node_partial_order;
    std::tie(initializers, nodes, node_partial_order) =
        instant::load_onnx("../data/VGG.onnx");
    for (auto const& p : initializers) {
        std::cout << p.first << std::endl;
        for (auto d : p.second.dims()) {
            std::cout << d << " ";
        }
        std::cout << "\n";
    }
    for (auto const& node_set : node_partial_order) {
        for (auto const* node : node_set) {
            std::cout << node << " ";
        }
        std::cout << "\n";
    }
}
*/
TEST_F(ONNXTest, load_onnx_model) {
    auto batch_size = 1;
    auto onnx_model = instant::load_onnx("../data/VGG.onnx");
    auto parameter_table = make_parameter_table(onnx_model.graph());
    std::unordered_map<std::string, instant::array> variable_table;
    variable_table["gpu_0/data_0"] = instant::array(
        instant::dtype_t::float_, {batch_size, 3, 224, 224},
        std::unique_ptr<float[]>(new float[batch_size * 3 * 224 * 224]));
    for (auto const& p : parameter_table) {
        std::cout << p.first << " " << p.second.data() << std::endl;
    }
}

}  // namespace
}  // namespace instant
