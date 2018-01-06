#ifndef INSTANT_MKLDNN_OPERATOR_CONV
#define INSTANT_MKLDNN_OPERATOR_CONV

#include <optional>
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

namespace instant::mkldnn_backend {

    inline auto make_conv_primitive(
      instant::node const& node,
      std::unordered_map<std::string, array> const& parameter_table,
      std::unordered_map<std::string, mkldnn::memory> const&
        variable_memory_table,
      std::set<std::string> const& required_output_name_set,
      mkldnn::engine const& engine) {
        std::cout << "make_conv_primitive" << std::endl;
        std::vector<mkldnn::primitive> net;
        std::unordered_map<std::string, mkldnn::memory> output_memory_table;
        std::unordered_map<std::string, array> output_table;
        std::vector<mkldnn::memory> temp_memory_list;

        auto const & [ strides, kernel_shape, pads ] =
          attributes_for_2d_data_processing(node);
        std::vector<int> padding_l{pads[0], pads[2]};
        std::vector<int> padding_r{pads[1], pads[3]};
        std::cout << "here" << std::endl;

        auto const& input_memory =
          find_value(variable_memory_table, node.input(0));
        auto weight_memory =
          array_to_memory(find_value(parameter_table, node.input(1)),
                          mkldnn::memory::format::oihw, engine);
        temp_memory_list.push_back(weight_memory);

        auto input_dims = extract_dims(input_memory);
        auto weight_dims = extract_dims(weight_memory);
        auto output_dims = calc_2d_output_dims(input_dims, weight_dims[0],
                                               kernel_shape, strides, pads);

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

        std::optional<mkldnn::convolution_forward::desc> conv_desc_opt;
        std::optional<mkldnn::memory> bias_memory_opt;
        if(node.input().size() == 2) {
            conv_desc_opt = mkldnn::convolution_forward::desc(
              mkldnn::prop_kind::forward_inference,
              mkldnn::algorithm::convolution_direct, conv_input_md,
              conv_weight_md, conv_output_md, strides, padding_l, padding_r,
              mkldnn::padding_kind::zero);
        } else {
            bias_memory_opt =
              array_to_memory(find_value(parameter_table, node.input(2)),
                              mkldnn::memory::format::x, engine);
            temp_memory_list.push_back(*bias_memory_opt);
            conv_desc_opt = mkldnn::convolution_forward::desc(
              mkldnn::prop_kind::forward_inference,
              mkldnn::algorithm::convolution_direct, conv_input_md,
              conv_weight_md, bias_memory_opt->get_primitive_desc().desc(),
              conv_output_md, strides, padding_l, padding_r,
              mkldnn::padding_kind::zero);
        }
        auto conv_pd =
          mkldnn::convolution_forward::primitive_desc(*conv_desc_opt, engine);

        auto conv_input_memory = input_memory;
        if(mkldnn::memory::primitive_desc(conv_pd.src_primitive_desc()) !=
           input_memory.get_primitive_desc()) {
            conv_input_memory = mkldnn::memory(conv_pd.src_primitive_desc());
            temp_memory_list.push_back(conv_input_memory);
            net.push_back(mkldnn::reorder(input_memory, conv_input_memory));
        }

        auto conv_weight_memory = weight_memory;
        if(mkldnn::memory::primitive_desc(conv_pd.weights_primitive_desc()) !=
           weight_memory.get_primitive_desc()) {
            conv_weight_memory =
              mkldnn::memory(conv_pd.weights_primitive_desc());
            temp_memory_list.push_back(conv_weight_memory);
            net.push_back(mkldnn::reorder(weight_memory, conv_weight_memory));
        }

        manage_output_memory(
          net, required_output_name_set, output_name, dtype_t::float_,
          output_dims, mkldnn::memory::format::nchw,
          conv_pd.dst_primitive_desc(), output_memory_table, output_table,
          temp_memory_list, engine,
          [&net, &conv_input_memory, &conv_weight_memory, &conv_pd,
           &bias_memory_opt](auto& op_output_memory) {
              if(bias_memory_opt) {
                  net.push_back(mkldnn::convolution_forward(
                    conv_pd, conv_input_memory, conv_weight_memory,
                    *bias_memory_opt, op_output_memory));
              } else {
                  net.push_back(mkldnn::convolution_forward(
                    conv_pd, conv_input_memory, conv_weight_memory,
                    op_output_memory));
              }
          });

        return std::make_tuple(net, output_memory_table, output_table,
                               temp_memory_list);
    }

} // namespace instant::mkldnn_backend

#endif // INSTANT_OPERATOR_CONV
