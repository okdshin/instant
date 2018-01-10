#ifndef INSTANT_MKLDNN_MODEL_HPP
#define INSTANT_MKLDNN_MODEL_HPP

#include <functional>
#include <iterator>
#include <unordered_map>

#include <mkldnn.hpp>

#include <instant/array.hpp>
#include <instant/context.hpp>
#include <instant/graph.hpp>

#include <instant/mkldnn/operator.hpp>
#include <instant/mkldnn/utility.hpp>

namespace instant::mkldnn_backend {

    using primitive_factory = std::function<std::tuple<
      std::vector<mkldnn::primitive>, // net
      std::unordered_map<std::string,
                         mkldnn::memory>,     // output_memory_table
      std::unordered_map<std::string, array>, // output_table
      std::vector<mkldnn::memory>             // temp_value_memory_list
      >(node,
        std::unordered_map<std::string,
                           array> const&, // parameter table
        std::unordered_map<std::string,
                           mkldnn::memory> const&, // variable memory table
        std::set<std::string> const&,              // required output name set
        mkldnn::engine const&)>;

    inline auto make_default_primitive_factory_table() {
        std::unordered_map<op_type_t, primitive_factory>
          primitive_factory_table;
        // primitive_factory_table.insert({"Abs", make_abs_primitive});
        /*
        primitive_factory_table.insert(
          {"AveragePool", make_average_pool_primitive});
        primitive_factory_table.insert(
          {"BatchNormalization", make_batch_norm_primitive});
        */
        primitive_factory_table.insert({op_type_t::conv, make_conv_primitive});
        primitive_factory_table.insert({op_type_t::fc, make_fc_primitive});
        primitive_factory_table.insert(
          {op_type_t::max_pool, make_max_pool_primitive});
        primitive_factory_table.insert({op_type_t::relu, make_relu_primitive});
        primitive_factory_table.insert(
          {op_type_t::softmax, make_softmax_primitive});
        /*
        primitive_factory_table.insert({"Dropout", make_dropout_primitive});
        primitive_factory_table.insert({"Elu", make_elu_primitive});
        primitive_factory_table.insert(
          {"LeakyRelu", make_leaky_relu_primitive});
        primitive_factory_table.insert({"Reshape", make_reshape_primitive});
        // primitive_factory_table.insert({"Sqrt", make_sqrt_primitive});
        primitive_factory_table.insert({"Tanh", make_tanh_primitive});
        // TODO other primitives
        */
        return primitive_factory_table;
    }

    inline auto make_nets(
      instant::graph const& graph,
      std::unordered_map<std::string, array> const& parameter_table,
      std::unordered_map<std::string, array> const& input_table,
      std::set<std::string> const& required_output_name_set,
      std::unordered_map<op_type_t, primitive_factory> primitive_factory_table =
        make_default_primitive_factory_table(),
      context const& context = instant::get_context()) {
        std::vector<mkldnn::primitive> nets;
        std::unordered_map<std::string, mkldnn::memory> variable_memory_table;
        for(auto const & [ name, arr ] : input_table) {
            mkldnn::memory::format format;
            if(arr.dims().size() == 2) {
                format = mkldnn::memory::format::nc;
            } else if(arr.dims().size() == 4) {
                format = mkldnn::memory::format::nchw;
            } else {
                throw std::runtime_error("Invalid input dims: " + name + " " +
                                         std::to_string(arr.dims().size()));
            }
            auto mem = array_to_memory(arr, format, context.engine());
            variable_memory_table.insert({name, mem});
        }
        std::unordered_map<std::string, instant::array> output_table;
        std::vector<mkldnn::memory> temp_value_memory_list;
        for(auto const& node_set : graph) {
            for(auto const& node : node_set) {
                try {
                    auto primitive_factory_pair_iter =
                      primitive_factory_table.find(node.op_type());
                    if(primitive_factory_pair_iter ==
                       primitive_factory_table.end()) {
                        throw std::runtime_error(
                          "Implementation not found: " +
                          op_type_to_string(node.op_type()));
                    }
                    auto[net, new_output_memory_table, new_output_table,
                         new_temp_value_memory_list] =
                      primitive_factory_pair_iter->second.operator()(
                        node, parameter_table, variable_memory_table,
                        required_output_name_set, context.engine());

                    nets.insert(nets.end(), net.begin(), net.end());
                    variable_memory_table.insert(
                      new_output_memory_table.begin(),
                      new_output_memory_table.end());
                    output_table.insert(new_output_table.begin(),
                                        new_output_table.end());
                    temp_value_memory_list.insert(
                      temp_value_memory_list.end(),
                      new_temp_value_memory_list.begin(),
                      new_temp_value_memory_list.end());
                } catch(mkldnn::error const& e) {
                    std::cout << "MKLDNN Error: " << e.message << std::endl;
                } catch(std::exception const& e) {
                    std::cout << "Error: " << e.what() << std::endl;
                }
            }
        }
        return std::make_tuple(nets, variable_memory_table,
                               temp_value_memory_list, output_table);
    }

} // namespace instant::mkldnn_backend
#endif // INSTANT_MODEL_HPP
