#ifndef INSTANT_OPERATOR_RESHAPE_HPP
#define INSTANT_OPERATOR_RESHAPE_HPP

#include <instant/operator/common.hpp>

namespace instant {

    inline auto calc_reshaped_dims(mkldnn::memory::dims const& base_shape,
                                   mkldnn::memory::dims new_shape) {
        // TODO for 0
        auto base_total_size = calc_total_size(base_shape);
        auto new_total_size = calc_total_size(new_shape);
        if(std::all_of(new_shape.begin(), new_shape.end(),
                       [](auto i) { return i > 0; })) {
            if(base_total_size != new_total_size) {
                throw std::runtime_error("invalid reshape");
            }
            return new_shape;
        }
        if(std::count(new_shape.begin(), new_shape.end(), -1) != 1) {
            throw std::runtime_error("invalid reshape (too many -1)");
        }
        if(base_total_size % new_total_size != 0) {
            throw std::runtime_error(
              "invalid reshape (cannot calc valid value for -1)");
        }
        auto div = -base_total_size / new_total_size;
        *std::find(new_shape.begin(), new_shape.end(), -1) = div;
        return new_shape;
    }

    inline auto make_reshape_primitive(
      std::unordered_map<std::string, const mkldnn::memory> const&
      /*parameter_memory_table*/,
      std::unordered_map<std::string, std::tuple<const mkldnn::memory,
                                                 mkldnn::memory::format>> const&
        variable_memory_table,
      std::set<std::string> const& /*required_output_set*/,
      onnx::NodeProto const& node, mkldnn::engine const& engine) {
        auto attribute_table = instant::make_attribute_table(node);
        auto shape = load_attribute_ints(attribute_table, "shape");

        auto const& input_memory_and_origin_format =
          find_value(variable_memory_table, node.input(0));
        auto const& input_memory = std::get<0>(input_memory_and_origin_format);
        auto input_dims = extract_dims(input_memory);
        shape[0] = input_dims[0];
        auto output_dims = calc_reshaped_dims(input_dims, shape);

        auto const& output_name = node.output(0);

        std::vector<mkldnn::memory>
          temp_variable_memory_list; // for temporary memory's life

        std::vector<mkldnn::primitive> net;

        auto op_input_memory = input_memory;
        if(input_memory.get_primitive_desc().desc().data.format !=
           mkldnn::memory::format::nchw) {
            op_input_memory = mkldnn::memory({{{input_dims},
                                               mkldnn::memory::data_type::f32,
                                               mkldnn::memory::format::nchw},
                                              engine});
            temp_variable_memory_list.push_back(op_input_memory);
            net.push_back(mkldnn::reorder(input_memory, op_input_memory));
        }

        // TODO manage output
        auto op_output_memory =
          mkldnn::memory({{{output_dims},
                           mkldnn::memory::data_type::f32,
                           mkldnn::memory::format::nc},
                          engine},
                         op_input_memory.get_data_handle());

        auto output_name_and_mem_and_origin_format = std::make_pair(
          output_name, std::make_tuple(std::move(op_output_memory),
                                       mkldnn::memory::format::nc));
        return std::make_tuple(
          net,
          std::vector<decltype(output_name_and_mem_and_origin_format)>{
            std::move(output_name_and_mem_and_origin_format)},
          temp_variable_memory_list,
          std::vector<std::pair<std::string, array>>());
    }

} // namespace instant

#endif // INSTANT_OPERATOR_RESHAPE_HPP
