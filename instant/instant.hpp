#ifndef INSTANT_INSTANT_HPP
#define INSTANT_INSTANT_HPP

#include <instant/model.hpp>

namespace instant {

    class model {
    public:
        model(onnx::ModelProto const& onnx_model,
              std::unordered_map<std::string, array> const& parameter_table,
              std::vector<array> const& temp_array_list,
              std::unordered_map<std::string, const mkldnn::memory> const&
                parameter_memory_table,
              std::unordered_map<std::string, array> const& input_table,
              std::unordered_map<
                std::string,
                std::tuple<const mkldnn::memory, mkldnn::memory::format>> const&
                input_memory_table,
              std::unordered_map<std::string, array> const& output_table,
              std::vector<mkldnn::primitive> const& nets,
              std::unordered_map<
                std::string,
                std::tuple<const mkldnn::memory, mkldnn::memory::format>> const&
                variable_memory_table,
              std::vector<mkldnn::memory> const& temp_variable_memory_list)
          : onnx_model_(onnx_model), parameter_table_(parameter_table),
            temp_array_list_(temp_array_list),
            parameter_memory_table_(parameter_memory_table),
            input_table_(input_table), input_memory_table_(input_memory_table),
            output_table_(output_table), nets_(nets),
            variable_memory_table_(variable_memory_table),
            temp_variable_memory_list_(temp_variable_memory_list) {}

        auto& input(std::string const& input_name) {
            return find_value(input_table_, input_name);
        }
        auto const& output(std::string const& input_name) const {
            return find_value(output_table_, input_name);
        }

        auto const& run() const {
            mkldnn::stream(mkldnn::stream::kind::eager).submit(nets_).wait();
            return output_table_;
        }

    private:
        onnx::ModelProto onnx_model_;
        std::unordered_map<std::string, array> parameter_table_;
        std::vector<array> temp_array_list_;
        std::unordered_map<std::string, const mkldnn::memory>
          parameter_memory_table_;
        std::unordered_map<std::string, array> input_table_;
        std::unordered_map<
          std::string, std::tuple<const mkldnn::memory, mkldnn::memory::format>>
          input_memory_table_;
        std::unordered_map<std::string, array> output_table_;
        std::vector<mkldnn::primitive> nets_;
        std::unordered_map<
          std::string, std::tuple<const mkldnn::memory, mkldnn::memory::format>>
          variable_memory_table_;
        std::vector<mkldnn::memory> temp_variable_memory_list_;
    };

    inline auto make_model(
      onnx::ModelProto const& onnx_model,
      std::vector<std::tuple<std::string, dtype_t, std::vector<int> const&,
                             mkldnn::memory::format>> const&
        input_name_dtype_dims_format_list,
      std::vector<std::string> const& required_output_name_list,
      mkldnn::engine const& engine = ::instant::get_context().engine()) {
        auto parameter_table = make_parameter_table(onnx_model.graph());
        auto parameter_memory_table_and_temp_array_list =
          make_parameter_memory_table(onnx_model.graph(), parameter_table,
                                      engine);
        auto& parameter_memory_table =
          std::get<0>(parameter_memory_table_and_temp_array_list);
        auto& temp_array_list =
          std::get<1>(parameter_memory_table_and_temp_array_list);

        std::unordered_map<std::string, array> input_table;
        std::vector<std::tuple<std::string, array, mkldnn::memory::format>>
          input_list;
        for(auto const& input_name_dtype_dims_format :
            input_name_dtype_dims_format_list) {
            auto name = std::get<0>(input_name_dtype_dims_format);
            auto dtype = std::get<1>(input_name_dtype_dims_format);
            auto dims = std::get<2>(input_name_dtype_dims_format);
            auto format = std::get<3>(input_name_dtype_dims_format);
            auto arr = array(dtype, dims);
            input_table.insert({name, arr});
            input_list.push_back(std::make_tuple(name, arr, format));
        }
        auto input_memory_table =
          make_variable_memory_table(input_list, engine);
        auto temp_tuple = make_nets(
          onnx_model.graph(), parameter_memory_table, input_memory_table,
          std::set<std::string>(required_output_name_list.begin(),
                                required_output_name_list.end()));
        auto const& nets = std::get<0>(temp_tuple);
        auto const& variable_memory_table = std::get<1>(temp_tuple);
        auto const& temp_variable_memory_list = std::get<2>(temp_tuple);
        auto const& output_table = std::get<3>(temp_tuple);
        return model(onnx_model, parameter_table, temp_array_list,
                     parameter_memory_table, input_table, input_memory_table,
                     output_table, nets, variable_memory_table,
                     temp_variable_memory_list);
    }

} // namespace instant

#endif // INSTANT_INSTANT_HPP
