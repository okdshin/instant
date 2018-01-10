#ifndef INSTANT_MKLDNN_OPERATOR_SOFTMAX_HPP
#define INSTANT_MKLDNN_OPERATOR_SOFTMAX_HPP

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

    inline auto make_softmax_primitive(
      instant::node const& node,
      [[maybe_unused]] std::unordered_map<std::string, array> const& parameter_table,
      std::unordered_map<std::string, mkldnn::memory> const&
        variable_memory_table,
      std::set<std::string> const& required_output_name_set,
      mkldnn::engine const& engine) {

        constexpr auto softmax_axis = 1;

        std::vector<mkldnn::primitive> net;
        std::unordered_map<std::string, mkldnn::memory> output_memory_table;
        std::unordered_map<std::string, array> output_table;
        std::vector<mkldnn::memory> temp_memory_list;

        auto const& input_memory =
          find_value(variable_memory_table, node.input(0));

        auto input_dims = extract_dims(input_memory);
        auto output_dims = input_dims;

        auto const& output_name = node.output(0);

        auto op_desc = mkldnn::softmax_forward::desc(
          mkldnn::prop_kind::forward_inference,
          input_memory.get_primitive_desc().desc(), softmax_axis);
        auto op_pd = mkldnn::softmax_forward::primitive_desc(op_desc, engine);

        manage_output_memory(
          net, required_output_name_set, output_name, dtype_t::float_,
          output_dims, mkldnn::memory::format::nc,
          input_memory.get_primitive_desc(), output_memory_table, output_table,
          temp_memory_list, engine,
          [&net, &input_memory, &op_pd](auto& op_output_memory) {
              net.push_back(
                mkldnn::softmax_forward(op_pd, input_memory, op_output_memory));
          });

        return std::make_tuple(net, output_memory_table, output_table,
                               temp_memory_list);
    }

} // namespace instant::mkldnn_backend

#endif // INSTANT_MKLDNN_OPERATOR_SOFTMAX_HPP
