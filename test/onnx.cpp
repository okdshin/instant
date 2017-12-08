#include <gtest/gtest.h>
#include <onnx.pb.h>
#include <iostream>
#include <fstream>
#include <numeric>
#include <string>
#include <instant/load_onnx.hpp>

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>

namespace instant {
namespace {

class ONNXTest : public ::testing::Test {};

#ifdef AAAA
TEST_F(ONNXTest, load_onnx_model_and_parse) {
    namespace gpio = ::google::protobuf::io;

    std::ifstream ifs("../data/VGG.onnx");
    gpio::IstreamInputStream iis(&ifs);
    gpio::CodedInputStream cis(&iis);
    cis.SetTotalBytesLimit(std::numeric_limits<int>::max(), std::numeric_limits<int>::max());

    onnx::ModelProto model;
    if(!model.ParseFromCodedStream(&cis)) {
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
    for(auto const& tensor : graph.initializer()) {
        std::cerr << tensor.name() << std::endl;
        for(int j = 0; j < tensor.dims_size(); ++j) {
            std::cerr << tensor.dims(j) << " ";
        }
        std::cerr << "\n";

    }
    for(int i = 0; i < graph.node_size(); ++i) {
        auto node = graph.node(i);
        std::cerr << i << ":" << node.op_type() << std::endl;
        for(int j = 0; j < node.input_size(); ++j) {
            std::cerr << "\ti" << j << ":" << node.input(j) << std::endl;
            std::cerr << "\ti" << j << ":" << *node.mutable_input(j) << std::endl;
        }
        for(int j = 0; j < node.output_size(); ++j) {
            std::cerr << "\to" << j << ":" << node.output(j) << std::endl;
        }
        for(int j = 0; j < node.attribute_size(); ++j) {
            auto attribute = node.attribute(j);
            std::cerr << "\ta" << j << ":" << attribute.name() << AttributeProto_AttributeType_Name(attribute.type()) << std::endl;
            if (attribute.has_f()) {
                std::cerr << "\t\tf" << ":" << attribute.f() << std::endl;
            }
            if (attribute.has_i()) {
                std::cerr << "\t\ti" << ":" << attribute.i() << std::endl;
            }
            if (attribute.has_s()) {
                std::cerr << "\t\ts" << ":" << attribute.s() << std::endl;
            }
            if (attribute.floats_size()) {
                for(int k = 0; k < attribute.floats_size(); ++k) {
                    std::cerr << "\t\tfs" << attribute.floats(k) << " ";
                }
                std::cerr << "\n";
            }
            if (attribute.ints_size()) {
                for(int k = 0; k < attribute.ints_size(); ++k) {
                    std::cerr << "\t\tis" << attribute.ints(k) << " ";
                }
                std::cerr << "\n";
            }
            if (attribute.strings_size()) {
                for(int k = 0; k < attribute.strings_size(); ++k) {
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

TEST_F(ONNXTest, load_onnx_model) {
    auto initializers_and_serialized_nodes = instant::load_onnx("../data/VGG.onnx");
    auto initializers = std::get<0>(initializers_and_serialized_nodes);
    auto nodes = std::get<1>(initializers_and_serialized_nodes);
    for (auto&& p : initializers) {
        std::cout << p.first << std::endl;
        for(auto d : p.second.dims()) {
            std::cout << d << " ";
        }
        std::cout << "\n";

        /*
        auto total_size = std::accumulate(p.second.dims().begin(), p.second.dims().end(), 1, std::multiplies<int>());
        for(auto i = 0; i < total_size; ++i) {
            std::cout << *(static_cast<float*>(p.second.data())+i) << " ";
        }
        std::cout << std::endl;
        */
    }
    for (auto const& node_set : nodes) {
        for (auto const& node : node_set) {
            std::cout << node << " ";
        }
        std::cout << "\n";
    }
}


} // namespace
} // namespace instant
