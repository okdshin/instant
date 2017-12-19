#ifndef INSTANT_OPERATOR_FC_HPP
#define INSTANT_OPERATOR_FC_HPP

#include <instant/operator/common.hpp>

namespace instant {

    inline auto make_fc_primitive(
      std::unordered_map<std::string, const mkldnn::memory> const&
        parameter_memory_table,
      std::unordered_map<std::string, std::tuple<const mkldnn::memory,
                                                 mkldnn::memory::format>> const&
        variable_memory_table,
      std::set<std::string> const& required_output_set,
      onnx::NodeProto const& node, mkldnn::engine const& engine) {

        auto attribute_table = instant::make_attribute_table(node);

        auto axis = load_attribute_int(attribute_table, "axis");
        assert(axis == 1);
        auto axis_w = load_attribute_int(attribute_table, "axis_w");
        assert(axis_w == 1);

        auto const& input_memory_and_origin_format =
          find_value(variable_memory_table, node.input(0));
        auto const& input_memory = std::get<0>(input_memory_and_origin_format);
        auto input_origin_format = std::get<1>(input_memory_and_origin_format);
        auto const& weight_memory =
          find_value(parameter_memory_table, node.input(1));
        auto const& bias_memory =
          find_value(parameter_memory_table, node.input(2));
        auto input_dims = extract_dims(input_memory);
        auto weight_dims = extract_dims(weight_memory);
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

        std::vector<mkldnn::primitive> net;
        std::vector<mkldnn::memory>
          temp_variable_memory_list; // for temporary memory's life

        auto fc_input_memory = input_memory;
        if(mkldnn::memory::primitive_desc(fc_pd.src_primitive_desc()) !=
           input_memory.get_primitive_desc()) {
            fc_input_memory = mkldnn::memory(fc_pd.src_primitive_desc());
            temp_variable_memory_list.push_back(fc_input_memory);
            net.push_back(mkldnn::reorder(input_memory, fc_input_memory));
        }

        auto fc_weight_memory = weight_memory;
        if(mkldnn::memory::primitive_desc(fc_pd.weights_primitive_desc()) !=
           weight_memory.get_primitive_desc()) {
            fc_weight_memory = mkldnn::memory(fc_pd.weights_primitive_desc());
            temp_variable_memory_list.push_back(fc_weight_memory);
            net.push_back(mkldnn::reorder(weight_memory, fc_weight_memory));
        }

        std::vector<std::pair<
          std::string, std::tuple<mkldnn::memory, mkldnn::memory::format>>>
          variable_memory_list;
        std::vector<std::pair<std::string, array>> output_name_and_arr_list;

        manage_output_memory(
          required_output_set, output_name, dtype_t::float_, output_dims,
          input_origin_format, fc_pd.dst_primitive_desc(), variable_memory_list,
          temp_variable_memory_list, output_name_and_arr_list, net, engine,
          [&fc_pd, &fc_input_memory, &fc_weight_memory,
           &bias_memory](auto& op_output_memory) {
              return mkldnn::inner_product_forward(
                fc_pd, fc_input_memory, fc_weight_memory, bias_memory,
                op_output_memory);
          });

        return std::make_tuple(net, variable_memory_list,
                               temp_variable_memory_list,
                               output_name_and_arr_list);
    }

} // namespace instant

#endif // INSTANT_OPERATOR_FC_HPP
