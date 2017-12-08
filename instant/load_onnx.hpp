#ifndef INSTANT_LOAD_ONNX_HPP
#define INSTANT_LOAD_ONNX_HPP

#include <algorithm>
#include <exception>
#include <functional>
#include <onnx.pb.h>
#include <string>
#include <tuple>
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

    inline auto load_initializers(onnx::GraphProto const& graph) {
        std::map<std::string, instant::array> initializers;
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
                std::copy(tensor.float_data().begin(),
                          tensor.float_data().end(),
                          static_cast<float_t*>(data.get()));
            } else {
                // TODO
                assert(!"not implemented");
            }
            initializers[tensor.name()] =
              instant::array(d, std::move(dims), std::move(data));
        }
        return initializers;
    }

    inline auto load_nodes(onnx::GraphProto const& graph) {
        std::map<std::string, onnx::NodeProto> nodes;
        for(int i = 0; i < graph.node_size(); ++i) {
            auto id = std::to_string(i);
            nodes.emplace(id, graph.node(i));
        }
        return nodes;
    }

    inline auto
    serialize_nodes(std::map<std::string, array> const& initializers,
                    std::map<std::string, onnx::NodeProto> const& node_dict) {
        std::vector<std::string> availables;
        std::transform(initializers.begin(), initializers.end(),
                       std::back_inserter(availables),
                       [](auto const& e) { return e.first; });
        std::vector<std::string> all_inputs, all_outputs;
        for(auto const& node : graph.node()) {
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

        std::map<std::string, std::reference_wrapper<const onnx::NodeProto>>
          node_dict;
        std::vector<std::string> left_nodes;
        for(int i = 0; i < graph.node_size(); ++i) {
            auto id = std::to_string(i);
            node_dict.emplace(id, graph.node(i));
            left_nodes.push_back(id);
        }

        std::vector<std::vector<std::string>> serialized_nodes;
        while(!left_nodes.empty()) {
            /*
            for(auto const& node : availables) {
                std::cout << node << " ";
            }
            std::cout << "\n";
            */
            std::vector<std::string> nodes;
            std::copy_if(
              left_nodes.begin(), left_nodes.end(), std::back_inserter(nodes),
              [&availables, &node_dict](auto const& node) {
                  std::vector<std::string> unavailable_inputs;
                  auto iter = node_dict.find(node);
                  assert(iter != node_dict.end());
                  auto const& n = iter->second.get();
                  std::vector<std::string> inputs(n.input().begin(),
                                                  n.input().end());
                  std::sort(inputs.begin(), inputs.end());
                  std::sort(availables.begin(), availables.end());
                  std::set_difference(inputs.begin(), inputs.end(),
                                      availables.begin(), availables.end(),
                                      std::back_inserter(unavailable_inputs));
                  return unavailable_inputs.empty();
              });
            for(auto const& node : nodes) {
                // std::cout << node << " " <<
                // node_dict.find(node)->second.get().input(0) << "\n";
                auto& np = node_dict.find(node)->second.get();
                availables.insert(availables.end(), np.output().begin(),
                                  np.output().end());
            }
            // std::cout <<"---\n";

            if(nodes.empty()) {
                throw onnx_load_error("Invalid node definition");
            }
            serialized_nodes.push_back(nodes);
            auto end_iter = std::remove_if(
              left_nodes.begin(),
              left_nodes.end(), [& nodes = nodes](auto& left_node) {
                  return std::any_of(
                    nodes.begin(), nodes.end(),
                    [&left_node = left_node](auto const& node_to_be_extracted) {
                        return left_node == node_to_be_extracted;
                    });
              });
            left_nodes.erase(end_iter, left_nodes.end());
        }
        return serialized_nodes;
    }

    inline auto load_onnx(std::string const& filename) {
        namespace gpio = ::google::protobuf::io;

        std::ifstream ifs(filename);
        gpio::IstreamInputStream iis(&ifs);
        gpio::CodedInputStream cis(&iis);
        cis.SetTotalBytesLimit(std::numeric_limits<int>::max(),
                               std::numeric_limits<int>::max());

        onnx::ModelProto model;
        if(!model.ParseFromCodedStream(&cis)) {
            throw onnx_load_error("ONNX parse error");
        }
        assert(model.has_graph());
        auto initializers = load_initializers(model.graph());
        auto node_dict_and_nodes = serialize_nodes(model.graph(), initializers);
        auto node_dict = std::get<0>(node_dict_and_nodes);
        auto nodes = std::get<1>(node_dict_and_nodes);
        return std::make_tuple(initializers, node_dict, nodes);
    }

} // namespace instant

#endif // INSTANT_LOAD_ONNX_HPP
