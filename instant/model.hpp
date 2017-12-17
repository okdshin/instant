#ifndef INSTANT_MODEL_HPP
#define INSTANT_MODEL_HPP

#include <functional>
#include <iterator>
#include <unordered_map>

#include <mkldnn.hpp>

#include <instant/array.hpp>
#include <instant/context.hpp>
#include <instant/operator.hpp>

namespace instant {

    inline auto make_parameter_memory_pair(
      onnx::NodeProto const& node, int param_index,
      mkldnn::memory::format format,
      std::unordered_map<std::string, instant::array>& parameter_table,
      mkldnn::engine const& engine) {
        auto const& name = node.input(param_index);
        auto& arr = find_value(parameter_table, name);
        mkldnn::memory::dims tz(arr.dims().begin(), arr.dims().end());
        auto mem = mkldnn::memory(
          {{{tz}, mkldnn::memory::data_type::f32, format}, engine}, arr.data());
        return std::make_pair(name, mem);
    }

    inline auto make_parameter_memory_table(
      onnx::GraphProto const& graph,
      std::unordered_map<std::string, instant::array>& parameter_table,
      mkldnn::engine const& engine) {
        std::unordered_map<std::string, const mkldnn::memory> memory_table;
        for(auto const& node : graph.node()) {
            if(node.op_type() == "Conv") {
                constexpr auto weight_index = 1;
                memory_table.insert(make_parameter_memory_pair(
                  node, weight_index, mkldnn::memory::format::oihw,
                  parameter_table, engine));

                if(node.input_size() != 2) {
                    constexpr auto bias_index = 2;
                    memory_table.insert(make_parameter_memory_pair(
                      node, bias_index, mkldnn::memory::format::x,
                      parameter_table, engine));
                }
            } else if(node.op_type() == "FC") {
                constexpr auto weight_index = 1;
                constexpr auto bias_index = 2;
                memory_table.insert(make_parameter_memory_pair(
                  node, weight_index,
                  mkldnn::memory::format::oi, // MEMO: is it correct? result is
                                              // correct...
                  parameter_table, engine));
                memory_table.insert(make_parameter_memory_pair(
                  node, bias_index, mkldnn::memory::format::x, parameter_table,
                  engine));
            } else {
                // TODO
                /*
                throw std::runtime_error("Not implemented yet: " +
                                         node.op_type());
                */
            }
        }
        return memory_table;
    }

    inline auto make_variable_memory_table(
      std::vector<std::tuple<std::string, instant::array,
                             mkldnn::memory::format>>& input_list,
      mkldnn::engine const& engine) {
        std::unordered_map<
          std::string, std::tuple<const mkldnn::memory, mkldnn::memory::format>>
          memory_table;
        for(auto& input : input_list) {
            auto& arr = std::get<1>(input);
            auto format = std::get<2>(input);
            mkldnn::memory::dims tz(arr.dims().begin(), arr.dims().end());
            auto mem = mkldnn::memory(
              {{{tz}, mkldnn::memory::data_type::f32, format}, engine},
              arr.data());
            auto const& name = std::get<0>(input);
            memory_table.insert({name, {mem, format}});
        }
        return memory_table;
    }

    using primitive_factory =
      std::function<
        std::tuple<
          std::vector<mkldnn::primitive>, // net
          std::vector<std::pair<
            std::string, std::tuple<mkldnn::memory,
                                    mkldnn::memory::format>>>, // output name
                                                               // and [memory
                                                               // and origin
                                                               // format] list
          std::vector<mkldnn::memory>, // temporary variable memory list
          std::vector<std::pair<std::string, array>>> // reqired output
                                                      // name and array
                                                      // list
        (std::unordered_map<
           std::string, const mkldnn::memory> const&, // parameter memory table
         std::unordered_map<
           std::string,
           std::tuple<const mkldnn::memory,
                      mkldnn::memory::format>> const&, // variable memory
                                                       // table
         std::set<std::string> const&, // required output name set
         onnx::NodeProto const&, mkldnn::engine const&)>;

    inline auto make_default_primitive_factory_table() {
        std::unordered_map<std::string, primitive_factory>
          primitive_factory_table;
        primitive_factory_table.insert({"Conv", make_conv_primitive});
        primitive_factory_table.insert({"Relu", make_relu_primitive});
        primitive_factory_table.insert({"MaxPool", make_max_pool_primitive});
        primitive_factory_table.insert({"Reshape", make_reshape_primitive});
        primitive_factory_table.insert({"FC", make_fc_primitive});
        primitive_factory_table.insert({"Dropout", make_dropout_primitive});
        primitive_factory_table.insert({"Softmax", make_softmax_primitive});
        // TODO other primitives
        return primitive_factory_table;
    }

    inline auto make_nets(
      onnx::GraphProto const& graph,
      std::unordered_map<std::string, const mkldnn::memory> const&
        parameter_memory_table,
      std::unordered_map<
        std::string, std::tuple<const mkldnn::memory, mkldnn::memory::format>>&
        input_memory_table,
      std::set<std::string> const& required_output_set,
      std::unordered_map<std::string, primitive_factory>
        primitive_factory_table =
          instant::make_default_primitive_factory_table(),
      instant::context const& context = instant::get_context()) {
        auto variable_memory_table = input_memory_table;
        std::unordered_map<std::string, instant::array> output_table;
        std::vector<mkldnn::primitive> nets;
        std::vector<mkldnn::memory> temp_variable_memory_list;
        for(auto const& node : graph.node()) {
            try {
                auto primitive_factory_pair_iter =
                  primitive_factory_table.find(node.op_type());
                if(primitive_factory_pair_iter ==
                   primitive_factory_table.end()) {
                    throw std::runtime_error("Implementation not found: " +
                                             node.op_type());
                }
                auto temp_tuple =
                  primitive_factory_pair_iter->second.operator()(
                    parameter_memory_table, variable_memory_table,
                    required_output_set, node, context.engine());
                auto& net = std::get<0>(temp_tuple);
                auto& output_name_and_memory_and_origin_format_list =
                  std::get<1>(temp_tuple);
                auto& temp_vars = std::get<2>(temp_tuple);
                auto& output_name_and_arr_list = std::get<3>(temp_tuple);

                nets.insert(nets.end(), std::make_move_iterator(net.begin()),
                            std::make_move_iterator(net.end()));
                variable_memory_table.insert(
                  std::make_move_iterator(
                    output_name_and_memory_and_origin_format_list.begin()),
                  std::make_move_iterator(
                    output_name_and_memory_and_origin_format_list.end()));
                temp_variable_memory_list.insert(
                  temp_variable_memory_list.end(),
                  std::make_move_iterator(temp_vars.begin()),
                  std::make_move_iterator(temp_vars.end()));
                output_table.insert(
                  std::make_move_iterator(output_name_and_arr_list.begin()),
                  std::make_move_iterator(output_name_and_arr_list.end()));
            } catch(mkldnn::error const& e) {
                std::cout << "MKLDNN Error: " << e.message << std::endl;
            } catch(std::exception const& e) {
                std::cout << "Error: " << e.what() << std::endl;
            }
        }
        return std::make_tuple(nets, variable_memory_table,
                               temp_variable_memory_list, output_table);
    }

    inline auto run_model(
      onnx::GraphProto const& graph,
      std::unordered_map<std::string, const mkldnn::memory> const&
        parameter_memory_table,
      std::unordered_map<
        std::string, std::tuple<const mkldnn::memory, mkldnn::memory::format>>&
        input_memory_table,
      std::set<std::string> const& required_output_set,
      std::unordered_map<std::string, primitive_factory>
        primitive_factory_table =
          instant::make_default_primitive_factory_table(),
      instant::context const& context = instant::get_context()) {
        auto temp_tuple =
          make_nets(graph, parameter_memory_table, input_memory_table,
                    required_output_set, primitive_factory_table, context);
        auto const& nets = std::get<0>(temp_tuple);
        auto const& output_table = std::get<3>(temp_tuple);
        mkldnn::stream(mkldnn::stream::kind::eager).submit(nets).wait();
        return output_table;
    }

} // namespace instant
#endif // INSTANT_MODEL_HPP
