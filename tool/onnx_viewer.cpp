#include "../external/cmdline.h"
#include <instant/load_onnx.hpp>

int main(int argc, char** argv) {
    if(argc == 1) {
        std::cout << "please set ONNX file path" << std::endl;
        return 0;
    }
    auto onnx_model_path = argv[1];
    auto onnx_model = instant::load_onnx(onnx_model_path);

    std::cout << "ONNX version is " << onnx_model.ir_version() << std::endl;
    std::cout << "domain is " << onnx_model.domain() << std::endl;
    std::cout << "model version is " << onnx_model.model_version() << std::endl;
    std::cout << "producer name is " << onnx_model.producer_name() << std::endl;
    std::cout << "producer version is " << onnx_model.producer_version() << std::endl;

    auto const& graph = onnx_model.graph();

    std::cout << "parameter list\n";
    for(auto const& tensor : graph.initializer()) {
        std::cout << "name: " << tensor.name() << " dims: ";
        for(int j = 0; j < tensor.dims_size(); ++j) {
            std::cout << tensor.dims(j) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    std::cout << "node list\n";
    std::cout << "node num is " << graph.node_size() << std::endl;
    for(int i = 0; i < graph.node_size(); ++i) {
        auto node = graph.node(i);
        std::cout << i << ":" << node.op_type() << std::endl;
        for(int j = 0; j < node.input_size(); ++j) {
            std::cout << "\tinput" << j << ": " << node.input(j) << std::endl;
        }
        for(int j = 0; j < node.output_size(); ++j) {
            std::cout << "\toutput" << j << ": " << node.output(j) << std::endl;
        }
        for(int j = 0; j < node.attribute_size(); ++j) {
            auto attribute = node.attribute(j);
            std::cout << "\tattribute" << j << ": " << attribute.name() << " ";
                      //<< "type: " << AttributeProto_AttributeType_Name(attribute.type()) << " ";
            if(attribute.has_f()) {
                std::cout << "float"
                          << ": " << attribute.f() << std::endl;
            }
            if(attribute.has_i()) {
                std::cout << "int"
                          << ": " << attribute.i() << std::endl;
            }
            if(attribute.has_s()) {
                std::cout << "string"
                          << ": " << attribute.s() << std::endl;
            }
            if(attribute.floats_size()) {
                std::cout << "floats: ";
                for(int k = 0; k < attribute.floats_size(); ++k) {
                    std::cout << attribute.floats(k) << " ";
                }
                std::cout << "\n";
            }
            if(attribute.ints_size()) {
                std::cout << "ints: ";
                for(int k = 0; k < attribute.ints_size(); ++k) {
                    std::cout << attribute.ints(k) << " ";
                }
                std::cout << "\n";
            }
            if(attribute.strings_size()) {
                std::cout << "strings: ";
                for(int k = 0; k < attribute.strings_size(); ++k) {
                    std::cout << attribute.strings(k) << " ";
                }
                std::cout << "\n";
            }
            /* TODO
            if (attribute.tensors_size()) {
                std::cout << "\t\ttensors";
            }
            */
        }
    }
}
