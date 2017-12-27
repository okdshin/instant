#ifndef INSTANT_OPERATOR_POOL_HPP
#define INSTANT_OPERATOR_POOL_HPP

#include <instant/operator/common.hpp>
#include <mkldnn.hpp>

namespace instant {

    template <mkldnn::algorithm pooling_alg>
    inline auto make_pool_primitive(
      std::unordered_map<std::string, const mkldnn::memory> const&
      /*parameter_memory_table*/,
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
        auto input_dims = extract_dims(input_memory);
        auto output_channel_num = input_dims[1];
        auto output_dims =
          make_conv_output_dims(input_dims, output_channel_num, kernel_shape,
                                strides, padding_l, padding_r);
        std::unique_ptr<mkldnn::memory> output_memory_p;
        std::unique_ptr<instant::array> output_arr_p;
        auto const& output_name = node.output(0);
        std::vector<mkldnn::memory>
          temp_variable_memory_list; // for temporary memory's life

        std::vector<std::pair<
          std::string, std::tuple<mkldnn::memory, mkldnn::memory::format>>>
          variable_memory_list;
        std::vector<std::pair<std::string, array>> output_name_and_arr_list;

        auto pool_output_md =
          mkldnn::memory::desc({output_dims}, mkldnn::memory::data_type::f32,
                               mkldnn::memory::format::any);

        auto pool_desc = mkldnn::pooling_forward::desc(
          mkldnn::prop_kind::forward, pooling_alg,
          input_memory.get_primitive_desc().desc(), pool_output_md, strides,
          kernel_shape, padding_l, padding_r, mkldnn::padding_kind::zero);
        auto pool_pd =
          mkldnn::pooling_forward::primitive_desc(pool_desc, engine);

        std::vector<mkldnn::primitive> net;

        auto pool_indices_memory =
          mkldnn::memory(pool_pd.workspace_primitive_desc());
        temp_variable_memory_list.push_back(pool_indices_memory);

        manage_output_memory(
          required_output_set, output_name, dtype_t::float_, output_dims,
          input_origin_format, pool_pd.dst_primitive_desc(),
          variable_memory_list, temp_variable_memory_list,
          output_name_and_arr_list, net, engine,
          [&input_memory, &pool_indices_memory,
           &pool_pd](auto& op_output_memory) {
              return mkldnn::pooling_forward(
                pool_pd, input_memory, op_output_memory, pool_indices_memory);
          });

        return std::make_tuple(net, variable_memory_list,
                               temp_variable_memory_list,
                               output_name_and_arr_list);
    }

    inline auto make_max_pool_primitive(
      std::unordered_map<std::string, const mkldnn::memory> const&
        parameter_memory_table,
      std::unordered_map<std::string, std::tuple<const mkldnn::memory,
                                                 mkldnn::memory::format>> const&
        variable_memory_table,
      std::set<std::string> const& required_output_set,
      onnx::NodeProto const& node, mkldnn::engine const& engine) {
        return make_pool_primitive<mkldnn::pooling_max>(
          parameter_memory_table, variable_memory_table, required_output_set,
          node, engine);
    }

    inline auto make_average_pool_primitive(
      std::unordered_map<std::string, const mkldnn::memory> const&
        parameter_memory_table,
      std::unordered_map<std::string, std::tuple<const mkldnn::memory,
                                                 mkldnn::memory::format>> const&
        variable_memory_table,
      std::set<std::string> const& required_output_set,
      onnx::NodeProto const& node, mkldnn::engine const& engine) {
        return make_pool_primitive<
          mkldnn::pooling_avg_include_padding>( // TODO check
          parameter_memory_table, variable_memory_table, required_output_set,
          node, engine);
    }

} // namespace instant

#endif // INSTANT_OPERATOR_POOL_HPP
