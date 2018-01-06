#ifndef INSTANT_OPERATOR_BATCH_NORM_HPP
#define INSTANT_OPERATOR_BATCH_NORM_HPP

#include <instant/operator/common.hpp>
#include <numeric>

namespace instant {

    inline auto make_batch_norm_primitive(
      std::unordered_map<std::string, const mkldnn::memory> const&
        parameter_memory_table,
      std::unordered_map<std::string, std::tuple<const mkldnn::memory,
                                                 mkldnn::memory::format>> const&
        variable_memory_table,
      std::set<std::string> const& required_output_set,
      onnx::NodeProto const& node, mkldnn::engine const& engine) {

        auto attribute_table = instant::make_attribute_table(node);

        auto epsilon = load_attribute_float(attribute_table, "epsilon");
        auto is_test =
          static_cast<bool>(load_attribute_int(attribute_table, "is_test"));
        if(!is_test) {
            throw std::runtime_error("Not implemented: is_test==false"); // TODO
        }
        auto spatial =
          static_cast<bool>(load_attribute_int(attribute_table, "spatial"));
        if(!spatial) {
            throw std::runtime_error("Not implemented: spatial==false"); // TODO
        }

        auto const& input_memory_and_origin_format =
          find_value(variable_memory_table, node.input(0));
        auto const& input_memory = std::get<0>(input_memory_and_origin_format);
        auto input_origin_format = std::get<1>(input_memory_and_origin_format);
        std::cout << "input origin format " << input_origin_format << std::endl;
        std::cout << "input actual format "
                  << input_memory.get_primitive_desc().desc().data.format
                  << std::endl;
        auto const& weights_memory =
          find_value(parameter_memory_table, node.input(1));
        /*
        auto const& b_memory =
          find_value(parameter_memory_table, node.input(2));
        */
        auto const& mean_memory =
          find_value(parameter_memory_table, node.input(3));
        auto const& var_memory =
          find_value(parameter_memory_table, node.input(4));
        auto input_output_dims = extract_dims(input_memory);
        auto scale_and_b_dims = [&weights_memory](){
            auto scale_and_b_dims = extract_dims(weights_memory);
            scale_and_b_dims.erase(scale_and_b_dims.begin());
            return scale_and_b_dims;
        }();

        /*
        std::cout << "eps " << epsilon << std::endl;
        std::cout << "src "
                  << *static_cast<float*>(input_memory.get_data_handle())
                  << std::endl;
        std::cout << "mean "
                  << *static_cast<float*>(mean_memory.get_data_handle())
                  << std::endl;
        std::cout << "var "
                  << *static_cast<float*>(var_memory.get_data_handle())
                  << std::endl;
        std::cout << "scale "
                  << *static_cast<float*>(weights_memory.get_data_handle())
                  << std::endl;
        std::cout << "b "
                  << *(static_cast<float*>(weights_memory.get_data_handle()) +
                       calc_total_size(scale_and_b_dims))
                  << std::endl;
        */

        auto mean_dims = extract_dims(mean_memory);
        auto var_dims = extract_dims(var_memory);
        auto c = input_output_dims[1];
        if(scale_and_b_dims[0] != c || mean_dims[0] != c || var_dims[0] != c) {
            throw std::runtime_error("invalid size");
        }

        auto const& output_name = node.output(0);

        mkldnn::batch_normalization_forward::desc bn_desc(
          mkldnn::prop_kind::forward_inference,
          input_memory.get_primitive_desc().desc(), epsilon,
          mkldnn::use_global_stats | mkldnn::use_scale_shift);
        auto bn_pd =
          mkldnn::batch_normalization_forward::primitive_desc(bn_desc, engine);

        std::vector<mkldnn::primitive> net;
        std::vector<mkldnn::memory>
          temp_variable_memory_list; // for temporary memory's life

        //TODO
        if(mkldnn::memory::primitive_desc(bn_pd.mean_primitive_desc()) !=
           mean_memory.get_primitive_desc()) {
            throw std::runtime_error("mean primitive is invalid");
        }
        //TODO
        if(mkldnn::memory::primitive_desc(bn_pd.variance_primitive_desc()) !=
           var_memory.get_primitive_desc()) {
            throw std::runtime_error("var primitive is invalid");
        }

        auto bn_weights_memory = weights_memory;
        if(mkldnn::memory::primitive_desc(bn_pd.weights_primitive_desc()) !=
           weights_memory.get_primitive_desc()) {
            bn_weights_memory = mkldnn::memory(bn_pd.weights_primitive_desc());
            temp_variable_memory_list.push_back(bn_weights_memory);
            net.push_back(mkldnn::reorder(weights_memory, bn_weights_memory));
        }

        std::vector<std::pair<
          std::string, std::tuple<mkldnn::memory, mkldnn::memory::format>>>
          variable_memory_list;
        std::vector<std::pair<std::string, array>> output_name_and_arr_list;

        manage_output_memory(
          required_output_set, output_name, dtype_t::float_, input_output_dims,
          input_origin_format, bn_pd.dst_primitive_desc(), variable_memory_list,
          temp_variable_memory_list, output_name_and_arr_list, net, engine,
          [&bn_pd, &input_memory, &bn_weights_memory, &mean_memory,
           &var_memory](auto& op_output_memory) {
              return mkldnn::batch_normalization_forward(
                bn_pd, static_cast<mkldnn::primitive::at>(input_memory),
                static_cast<mkldnn::primitive::at>(mean_memory),
                static_cast<mkldnn::primitive::at>(var_memory),
                static_cast<mkldnn::primitive::at>(bn_weights_memory),
                op_output_memory);
          });

        return std::make_tuple(net, variable_memory_list,
                               temp_variable_memory_list,
                               output_name_and_arr_list);
    }

} // namespace instant

#endif // INSTANT_OPERATOR_BATCH_NORM_HPP
