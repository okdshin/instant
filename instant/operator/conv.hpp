#ifndef INSTANT_OPERATOR_CONV
#define INSTANT_OPERATOR_CONV

#include <instant/operator/common.hpp>

namespace instant {

    inline auto make_conv_primitive(
      std::unordered_map<std::string, const mkldnn::memory> const&
        parameter_memory_table,
      std::unordered_map<std::string, std::tuple<const mkldnn::memory,
                                                 mkldnn::memory::format>> const&
        variable_memory_table,
      std::set<std::string> const& required_output_set,
      onnx::NodeProto const& node, mkldnn::engine const& engine) {

        auto attribute_table = instant::make_attribute_table(node);

        auto attributes = load_2d_data_processing_attributes(attribute_table);
        auto const& strides = std::get<0>(attributes);
        auto const& kernel_shape = std::get<1>(attributes);
        auto const& padding_l = std::get<2>(attributes);
        auto const& padding_r = std::get<3>(attributes);

        auto const& input_memory_and_origin_format =
          find_value(variable_memory_table, node.input(0));
        auto const& input_memory = std::get<0>(input_memory_and_origin_format);
        auto input_origin_format = std::get<1>(input_memory_and_origin_format);
        auto const& weight_memory =
          find_value(parameter_memory_table, node.input(1));

        auto input_dims = extract_dims(input_memory);
        auto weight_dims = extract_dims(weight_memory);
        auto output_dims =
          make_conv_output_dims(input_dims, weight_dims[0], kernel_shape,
                                strides, padding_l, padding_r);

        auto const& output_name = node.output(0);

        auto conv_input_md =
          mkldnn::memory::desc({input_dims}, mkldnn::memory::data_type::f32,
                               mkldnn::memory::format::any);
        auto conv_weight_md =
          mkldnn::memory::desc({weight_dims}, mkldnn::memory::data_type::f32,
                               mkldnn::memory::format::any);
        auto conv_output_md =
          mkldnn::memory::desc({output_dims}, mkldnn::memory::data_type::f32,
                               mkldnn::memory::format::any);

        std::unique_ptr<mkldnn::convolution_forward::desc> conv_desc_p;
        if(node.input_size() == 2) {
            conv_desc_p = std::make_unique<mkldnn::convolution_forward::desc>(
              mkldnn::prop_kind::forward_inference,
              mkldnn::algorithm::convolution_direct, conv_input_md,
              conv_weight_md, conv_output_md, strides, padding_l, padding_r,
              mkldnn::padding_kind::zero);
        } else {
            auto const& bias_memory =
              find_value(parameter_memory_table, node.input(2));
            conv_desc_p = std::make_unique<mkldnn::convolution_forward::desc>(
              mkldnn::prop_kind::forward_inference,
              mkldnn::algorithm::convolution_direct, conv_input_md,
              conv_weight_md, bias_memory.get_primitive_desc().desc(),
              conv_output_md, strides, padding_l, padding_r,
              mkldnn::padding_kind::zero);
        }
        auto conv_pd =
          mkldnn::convolution_forward::primitive_desc(*conv_desc_p, engine);

        std::vector<mkldnn::primitive> net;
        std::vector<mkldnn::memory>
          temp_variable_memory_list; // for temporary memory's life

        auto conv_input_memory = input_memory;
        if(mkldnn::memory::primitive_desc(conv_pd.src_primitive_desc()) !=
           input_memory.get_primitive_desc()) {
            conv_input_memory = mkldnn::memory(conv_pd.src_primitive_desc());
            temp_variable_memory_list.push_back(conv_input_memory);
            net.push_back(mkldnn::reorder(input_memory, conv_input_memory));
        }

        auto conv_weight_memory = weight_memory;
        if(mkldnn::memory::primitive_desc(conv_pd.weights_primitive_desc()) !=
           weight_memory.get_primitive_desc()) {
            conv_weight_memory =
              mkldnn::memory(conv_pd.weights_primitive_desc());
            temp_variable_memory_list.push_back(conv_weight_memory);
            net.push_back(mkldnn::reorder(weight_memory, conv_weight_memory));
        }

        std::vector<std::pair<
          std::string, std::tuple<mkldnn::memory, mkldnn::memory::format>>>
          variable_memory_list;
        std::vector<std::pair<std::string, array>> output_name_and_arr_list;

        manage_output_memory(
          required_output_set, output_name, dtype_t::float_, output_dims,
          input_origin_format, conv_pd.dst_primitive_desc(),
          variable_memory_list, temp_variable_memory_list,
          output_name_and_arr_list, net, engine,
          [&conv_input_memory, &conv_weight_memory, &node, &conv_pd,
           &parameter_memory_table](auto& op_output_memory) {
              if(node.input_size() == 2) {
                  return mkldnn::convolution_forward(conv_pd, conv_input_memory,
                                                     conv_weight_memory,
                                                     op_output_memory);
              } else {
                  auto const& conv_bias_memory =
                    find_value(parameter_memory_table, node.input(2));
                  return mkldnn::convolution_forward(
                    conv_pd, conv_input_memory, conv_weight_memory,
                    conv_bias_memory, op_output_memory);
              }
          });

        return std::make_tuple(net, variable_memory_list,
                               temp_variable_memory_list,
                               output_name_and_arr_list);
    }

} // namespace instant

#endif // INSTANT_OPERATOR_CONV
