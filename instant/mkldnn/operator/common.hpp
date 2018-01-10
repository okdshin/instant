#ifndef INSTANT_OPERATOR_COMMON_HPP
#define INSTANT_OPERATOR_COMMON_HPP

#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <mkldnn.hpp>

#include <instant/array.hpp>
#include <instant/context.hpp>
#include <instant/onnx.hpp>
#include <instant/utility.hpp>

#include <instant/mkldnn/utility.hpp>

namespace instant::mkldnn_backend {

    template <typename OpPrimitiveGenerator>
    auto manage_output_memory(
      std::vector<mkldnn::primitive>& net,
      std::set<std::string> const& required_output_set,
      std::string const& output_name, dtype_t output_dtype,
      std::vector<int> const& output_dims, mkldnn::memory::format output_format,
      mkldnn::memory::primitive_desc const& output_pd,
      std::unordered_map<std::string, mkldnn::memory>& output_memory_table,
      std::unordered_map<std::string, array>& output_table,
      std::vector<mkldnn::memory>& temp_memory_list,
      mkldnn::engine const& engine,
      OpPrimitiveGenerator op_primitive_generator) {

        for(int i = 0; i < output_dims.size(); ++i) {
            assert(output_dims[i] ==
                   const_cast<mkldnn::memory::primitive_desc&>(output_pd)
                     .desc()
                     .data.dims[i]);
        }

        std::optional<mkldnn::memory> output_memory_opt;
        std::optional<instant::array> output_arr_opt;

        if(required_output_set.find(output_name) != required_output_set.end()) {
            output_arr_opt = instant::array(output_dtype, output_dims);
            output_memory_opt =
              array_to_memory(*output_arr_opt, output_format, engine);
        }

        auto op_output_memory =
          output_memory_opt.value_or(mkldnn::memory(output_pd));
        if(output_memory_opt && mkldnn::memory::primitive_desc(output_pd) !=
                                  output_memory_opt->get_primitive_desc()) {
            op_output_memory = mkldnn::memory(output_pd);
            temp_memory_list.push_back(*output_memory_opt);
        }

        op_primitive_generator(op_output_memory);

        if(output_memory_opt && op_output_memory != *output_memory_opt) {
            net.push_back(
              mkldnn::reorder(op_output_memory, *output_memory_opt));
        }

        output_memory_table.insert({output_name, op_output_memory});
        if(output_arr_opt) {
            output_table.insert({output_name, *output_arr_opt});
        }
    }

} // namespace instant::mkldnn_backend

#endif // INSTANT_OPERATOR_COMMON_HPP
