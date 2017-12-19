#ifndef INSTANT_OPERATOR_DROPOUT_HPP
#define INSTANT_OPERATOR_DROPOUT_HPP

#include <instant/operator/common.hpp>
#include <instant/operator/nop.hpp>

namespace instant {

    inline auto make_dropout_primitive(
      std::unordered_map<std::string, const mkldnn::memory> const&
        parameter_memory_table,
      std::unordered_map<std::string, std::tuple<const mkldnn::memory,
                                                 mkldnn::memory::format>> const&
        variable_memory_table,
      std::set<std::string> const& required_output_set,
      onnx::NodeProto const& node, mkldnn::engine const& engine) {
        return make_nop_primitive(parameter_memory_table, variable_memory_table,
                                  required_output_set, node, engine);
    }

} // namespace instant

#endif // INSTANT_OPERATOR_DROPOUT_HPP
