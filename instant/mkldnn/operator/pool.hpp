#ifndef INSTANT_MKLDNN_OPERATOR_POOL_HPP
#define INSTANT_MKLDNN_OPERATOR_POOL_HPP

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

    template <mkldnn::algorithm pooling_alg>
    inline auto make_pool_primitive(
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

        auto const & [ strides, kernel_shape, pads ] =
          attributes_for_2d_data_processing(node);
        std::vector<int> padding_l{pads[0], pads[2]};
        std::vector<int> padding_r{pads[1], pads[3]};

        auto const& input_memory =
          find_value(variable_memory_table, node.input(0));

        auto input_dims = extract_dims(input_memory);
        auto output_channel_num = input_dims[1];
        auto output_dims = calc_2d_output_dims(input_dims, output_channel_num,
                                               kernel_shape, strides, pads);

        auto const& output_name = node.output(0);

        auto pool_output_md =
          mkldnn::memory::desc({output_dims}, mkldnn::memory::data_type::f32,
                               mkldnn::memory::format::any);
        auto pool_desc = mkldnn::pooling_forward::desc(
          mkldnn::prop_kind::forward, pooling_alg,
          input_memory.get_primitive_desc().desc(), pool_output_md, strides,
          kernel_shape, padding_l, padding_r, mkldnn::padding_kind::zero);
        auto pool_pd =
          mkldnn::pooling_forward::primitive_desc(pool_desc, engine);

        manage_output_memory(
          net, required_output_name_set, output_name, dtype_t::float_,
          output_dims, mkldnn::memory::format::nchw,
          pool_pd.dst_primitive_desc(), output_memory_table, output_table,
          temp_memory_list, engine, [
              pa = pooling_alg, &net, &input_memory, &temp_memory_list, &pool_pd
          ](auto& op_output_memory) {
              if(pa == mkldnn::pooling_max) {
                  auto pool_indices_memory =
                    mkldnn::memory(pool_pd.workspace_primitive_desc());
                  temp_memory_list.push_back(pool_indices_memory);
                  net.push_back(mkldnn::pooling_forward(pool_pd, input_memory,
                                                        op_output_memory,
                                                        pool_indices_memory));
              } else {
                  net.push_back(mkldnn::pooling_forward(pool_pd, input_memory,
                                                        op_output_memory));
              }
          });

        return std::make_tuple(net, output_memory_table, output_table,
                               temp_memory_list);
    }

    inline auto
    make_max_pool_primitive(instant::node const& node,
                  std::unordered_map<std::string, array> const& parameter_table,
                  std::unordered_map<std::string, mkldnn::memory> const&
                    variable_memory_table,
                  std::set<std::string> const& required_output_name_set,
                  mkldnn::engine const& engine) {
        return make_pool_primitive<mkldnn::pooling_max>(
          node, parameter_table, variable_memory_table,
          required_output_name_set, engine);
    }

    /*
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
    */

} // namespace instant::mkldnn_backend

#endif // INSTANT_MKLDNN_OPERATOR_POOL_HPP
