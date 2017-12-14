#ifndef INSTANT_LOAD_ONNX_HPP
#define INSTANT_LOAD_ONNX_HPP

#include <algorithm>
#include <exception>
#include <functional>
#include <numeric>
#include <onnx.pb.h>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <instant/array.hpp>
#include <instant/dtype.hpp>

namespace instant {

    class onnx_load_error : public std::runtime_error {
    public:
        onnx_load_error(std::string const& message) : runtime_error(message) {}
    };

    /*
    inline auto load_nodes(onnx::GraphProto const& graph) {
        std::vector<onnx::NodeProto> nodes;
        std::copy(graph.node().begin(), graph.node().end(),
    std::back_inserter(nodes)); return nodes;
    }

    inline auto
    serialize_nodes(std::map<std::string, array> const& initializers,
                    std::vector<onnx::NodeProto> const& nodes) {
        std::vector<std::string> availables;
        std::transform(initializers.begin(), initializers.end(),
                       std::back_inserter(availables),
                       [](auto const& e) { return e.first; });
        std::vector<std::string> all_inputs, all_outputs;
        for(auto const& node : nodes) {
            all_inputs.insert(all_inputs.end(), node.input().begin(),
                              node.input().end());
            all_outputs.insert(all_outputs.end(), node.output().begin(),
                               node.output().end());
        }
        std::sort(all_inputs.begin(), all_inputs.end());
        std::sort(all_outputs.begin(), all_outputs.end());
        std::vector<std::string> model_inputs;
        std::set_difference(all_inputs.begin(), all_inputs.end(),
                            all_outputs.begin(), all_outputs.end(),
                            std::back_inserter(model_inputs));
        availables.insert(availables.end(), model_inputs.begin(),
                          model_inputs.end());
        std::sort(availables.begin(), availables.end());

        std::vector<onnx::NodeProto const*> left_nodes;
        std::transform(nodes.begin(), nodes.end(),
                       std::back_inserter(left_nodes),
                       [](auto const& node) { return std::addressof(node); });

        std::vector<std::vector<onnx::NodeProto const*>> node_partial_order;
        while(!left_nodes.empty()) {
            std::vector<onnx::NodeProto const*> consumable_node_set;
            std::copy_if(
              left_nodes.begin(), left_nodes.end(),
              std::back_inserter(consumable_node_set), [&availables](auto* np) {
                  std::vector<std::string> unavailable_inputs;
                  std::vector<std::string> inputs(np->input().begin(),
                                                  np->input().end());
                  std::sort(inputs.begin(), inputs.end());
                  std::sort(availables.begin(), availables.end());
                  std::set_difference(inputs.begin(), inputs.end(),
                                      availables.begin(), availables.end(),
                                      std::back_inserter(unavailable_inputs));
                  return unavailable_inputs.empty();
              });
            for(auto* np : consumable_node_set) {
                // std::cout << np << " " <<
                // np->input(0) << "\n";
                availables.insert(availables.end(), np->output().begin(),
                                  np->output().end());
            }
            // std::cout <<"---\n";

            if(consumable_node_set.empty()) {
                throw onnx_load_error("Invalid node definition");
            }
            node_partial_order.push_back(consumable_node_set);
            auto end_iter = std::remove_if(
              left_nodes.begin(), left_nodes.end(),
              [&consumable_node_set](auto* left_node_ptr) {
                  return std::any_of(
                    consumable_node_set.begin(), consumable_node_set.end(),
                    [left_node_ptr](auto const* consumable_node) {
                        return left_node_ptr == consumable_node;
                    });
              });
            left_nodes.erase(end_iter, left_nodes.end());
        }
        return node_partial_order;
    }
    */

    inline auto load_onnx(std::string const& filename) {
        namespace gpio = ::google::protobuf::io;

        std::ifstream ifs(filename);
        gpio::IstreamInputStream iis(&ifs);
        gpio::CodedInputStream cis(&iis);
        cis.SetTotalBytesLimit(std::numeric_limits<int>::max(),
                               std::numeric_limits<int>::max());

        onnx::ModelProto onnx_model;
        if(!onnx_model.ParseFromCodedStream(&cis)) {
            throw onnx_load_error("ONNX parse error");
        }
        return onnx_model;
    }

    //TODO avoid copy
    /*
    inline auto make_parameter_table(onnx::GraphProto const& graph) {
        std::unordered_map<std::string,
                           std::reference_wrapper<const onnx::TensorProto>>
          parameter_table;
        for(auto const& initializer : graph.initializer()) {
            parameter_table.insert(
              {initializer.name(), std::cref(initializer)});
        }
        return parameter_table;
    }
    */

    inline auto make_parameter_table(onnx::GraphProto const& graph) {
        std::unordered_map<std::string, instant::array> parameter_table;
        for(int i = 0; i < graph.initializer_size(); ++i) {
            auto tensor = graph.initializer(i);
            assert(tensor.has_data_type());
            dtype_t d = tensor_proto_data_type_to_dtype_t(tensor.data_type());

            std::vector<int> dims(tensor.dims().begin(), tensor.dims().end());
            auto total_size = std::accumulate(dims.begin(), dims.end(), 1,
                                              std::multiplies<int>());

            std::shared_ptr<void> data;
            if(d == instant::dtype_t::float_) {
                using float_t =
                  instant::dtype_t_to_type_t<instant::dtype_t::float_>;
                data = std::unique_ptr<float_t[]>(new float_t[total_size]);
                // TODO other format: float_data
                assert(tensor.has_raw_data());
                assert(tensor.raw_data().length() == total_size*4);
                std::copy(tensor.raw_data().begin(),
                          tensor.raw_data().end(),
                          static_cast<char*>(data.get()));
            } else {
                throw onnx_load_error("Not implemented");
            }
            parameter_table.insert(
              {tensor.name(),
               instant::array(d, std::move(dims), std::move(data))});
        }
        return parameter_table;
    }

    inline auto make_attribute_table(onnx::NodeProto const& node) {
        std::unordered_map<std::string,
                           std::reference_wrapper<const onnx::AttributeProto>>
          attribute_table;
        for(auto const& attr : node.attribute()) {
            attribute_table.insert({attr.name(), std::cref(attr)});
        }
        return attribute_table;
    }

} // namespace instant

#endif // INSTANT_LOAD_ONNX_HPP
