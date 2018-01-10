#ifndef INSTANT_MKLDNN_OPERATOR_FC_HPP
#define INSTANT_MKLDNN_OPERATOR_FC_HPP

#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <mkldnn.hpp>

#include <instant/array.hpp>
#include <instant/graph.hpp>
#include <instant/utility.hpp>

#include <instant/mkldnn/operator/common.hpp>
#include <instant/mkldnn/utility.hpp>

namespace instant::mkldnn_backend {

    inline auto make_fc_primitive(
      instant::node const& node,
      [[maybe_unused]] std::unordered_map<std::string, array> const&
        parameter_table,
      std::unordered_map<std::string, mkldnn::memory> const&
        variable_memory_table,
      std::set<std::string> const& required_output_name_set,
      mkldnn::engine const& engine) {

        std::vector<mkldnn::primitive> net;
        std::unordered_map<std::string, mkldnn::memory> output_memory_table;
        std::unordered_map<std::string, array> output_table;
        std::vector<mkldnn::memory> temp_memory_list;

        auto axis = attribute_int(node, "axis");
        if(axis != 1) {
            throw std::runtime_error("axis must be 1: " + std::to_string(axis));
        }
        auto axis_w = attribute_int(node, "axis_w");
        if(axis_w != 1) {
            throw std::runtime_error("axis_w must be 1: " +
                                     std::to_string(axis_w));
        }

        auto const& input_memory =
          find_value(variable_memory_table, node.input(0));
        auto input_dims = extract_dims(input_memory);
        auto weight_format = input_dims.size() == 2
                               ? mkldnn::memory::format::oi
                               : mkldnn::memory::format::oihw;

        auto weight_arr = find_value(parameter_table, node.input(1));
        auto weight_dims = weight_arr.dims();
        if(weight_format == mkldnn::memory::format::oihw) {
            weight_dims = std::vector<int>{weight_dims.front()};
            weight_dims.insert(weight_dims.end(), input_dims.begin() + 1,
                               input_dims.end());
        }

        auto weight_memory =
          array_to_memory(weight_arr, weight_dims, weight_format, engine);
        temp_memory_list.push_back(weight_memory);
        auto bias_memory =
          array_to_memory(find_value(parameter_table, node.input(2)),
                          mkldnn::memory::format::x, engine);
        temp_memory_list.push_back(bias_memory);

        // auto weight_dims = extract_dims(weight_memory);
        auto bias_dims = extract_dims(bias_memory);
        mkldnn::memory::dims output_dims{input_dims[0], bias_dims[0]};

        auto const& output_name = node.output(0);

        auto fc_input_md =
          mkldnn::memory::desc({input_dims}, mkldnn::memory::data_type::f32,
                               mkldnn::memory::format::any);
        auto fc_weight_md =
          mkldnn::memory::desc({weight_dims}, mkldnn::memory::data_type::f32,
                               mkldnn::memory::format::any);
        auto fc_output_md =
          mkldnn::memory::desc({output_dims}, mkldnn::memory::data_type::f32,
                               mkldnn::memory::format::any);

        mkldnn::inner_product_forward::desc fc_desc(
          mkldnn::prop_kind::forward_inference, fc_input_md, fc_weight_md,
          bias_memory.get_primitive_desc().desc(), fc_output_md);
        auto fc_pd =
          mkldnn::inner_product_forward::primitive_desc(fc_desc, engine);

        auto fc_input_memory = input_memory;
        if(mkldnn::memory::primitive_desc(fc_pd.src_primitive_desc()) !=
           input_memory.get_primitive_desc()) {
            fc_input_memory = mkldnn::memory(fc_pd.src_primitive_desc());
            temp_memory_list.push_back(fc_input_memory);
            net.push_back(mkldnn::reorder(input_memory, fc_input_memory));
        }

        auto fc_weight_memory = weight_memory;
        if(mkldnn::memory::primitive_desc(fc_pd.weights_primitive_desc()) !=
           weight_memory.get_primitive_desc()) {
            fc_weight_memory = mkldnn::memory(fc_pd.weights_primitive_desc());
            temp_memory_list.push_back(fc_weight_memory);
            net.push_back(mkldnn::reorder(weight_memory, fc_weight_memory));
        }

        std::vector<std::pair<
          std::string, std::tuple<mkldnn::memory, mkldnn::memory::format>>>
          variable_memory_list;
        std::vector<std::pair<std::string, array>> output_name_and_arr_list;

        manage_output_memory(
          net, required_output_name_set, output_name, dtype_t::float_,
          output_dims, mkldnn::memory::format::nc, fc_pd.dst_primitive_desc(),
          output_memory_table, output_table, temp_memory_list, engine,
          [&net, &fc_input_memory, &fc_weight_memory, &fc_pd,
           &bias_memory](auto& op_output_memory) {
              net.push_back(mkldnn::inner_product_forward(
                fc_pd, fc_input_memory, fc_weight_memory, bias_memory,
                op_output_memory));
          });

        return std::make_tuple(net, output_memory_table, output_table,
                               temp_memory_list);
    }

} // namespace instant::mkldnn_backend

#endif // INSTANT_MKLDNN_OPERATOR_FC_HPP
