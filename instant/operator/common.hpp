#ifndef INSTANT_OPERATOR_COMMON_HPP
#define INSTANT_OPERATOR_COMMON_HPP

#include <string>
#include <unordered_map>

#include <mkldnn.hpp>

#include <instant/array.hpp>
#include <instant/context.hpp>
#include <instant/onnx.hpp>
#include <instant/utility.hpp>

namespace instant {

    inline auto extract_dims(mkldnn::memory const& m) {
        auto const& d = m.get_primitive_desc().desc().data;
        return std::vector<int>(d.dims, d.dims + d.ndims);
    }

    inline auto load_attribute_ints(
      std::unordered_map<
        std::string, std::reference_wrapper<const onnx::AttributeProto>> const&
        attribute_table,
      std::string const& attribute_name) {
        onnx::AttributeProto const& attr =
          find_value(attribute_table, attribute_name);
        if(attr.ints_size() == 0) {
            throw std::runtime_error(
              "Attribute load error: not ints attribute");
        }
        return std::vector<int>(attr.ints().begin(), attr.ints().end());
    }

    inline auto load_attribute_int(
      std::unordered_map<
        std::string, std::reference_wrapper<const onnx::AttributeProto>> const&
        attribute_table,
      std::string const& attribute_name) {
        onnx::AttributeProto const& attr =
          find_value(attribute_table, attribute_name);
        if(!attr.has_i()) {
            throw std::runtime_error("Attribute load error: not int attribute");
        }
        return attr.i();
    }

    inline auto load_attribute_float(
      std::unordered_map<
        std::string, std::reference_wrapper<const onnx::AttributeProto>> const&
        attribute_table,
      std::string const& attribute_name) {
        onnx::AttributeProto const& attr =
          find_value(attribute_table, attribute_name);
        if(!attr.has_f()) {
            throw std::runtime_error(
              "Attribute load error: not float attribute");
        }
        return attr.f();
    }

    inline auto load_2d_data_processing_attributes(
      std::unordered_map<
        std::string, std::reference_wrapper<const onnx::AttributeProto>> const&
        attribute_table) {
        auto strides = load_attribute_ints(attribute_table, "strides");
        assert(strides.size() == 2);

        auto kernel_shape =
          load_attribute_ints(attribute_table, "kernel_shape");
        assert(kernel_shape.size() == 2);

        onnx::AttributeProto const& pads_attr =
          find_value(attribute_table, "pads");
        mkldnn::memory::dims padding_l, padding_r;
        if(pads_attr.ints().size() == 4) {
            padding_l = mkldnn::memory::dims(pads_attr.ints().begin() + 0,
                                             pads_attr.ints().begin() + 2);
            padding_r = mkldnn::memory::dims(pads_attr.ints().begin() + 2,
                                             pads_attr.ints().begin() + 4);
        } else if(pads_attr.ints().size() == 2) {
            padding_l = padding_r = mkldnn::memory::dims(
              pads_attr.ints().begin() + 0, pads_attr.ints().begin() + 2);
        } else {
            throw std::runtime_error(
              "Not implemented pads size: " +
              std::to_string(pads_attr.ints().size())); // TODO
        }

        return std::make_tuple(strides, kernel_shape, padding_l, padding_r);
    }

    template <typename OpPrimitiveGenerator>
    auto manage_output_memory(
      std::set<std::string> const& required_output_set,
      std::string const& output_name, dtype_t output_dtype,
      std::vector<int> const& output_dims, mkldnn::memory::format output_format,
      mkldnn::memory::primitive_desc const& output_pd,
      std::vector<std::pair<
        std::string, std::tuple<mkldnn::memory, mkldnn::memory::format>>>&
        variable_memory_list,
      std::vector<mkldnn::memory>& temp_variable_memory_list,
      std::vector<std::pair<std::string, array>>& output_name_and_arr_list,
      std::vector<mkldnn::primitive>& net, mkldnn::engine const& engine,
      OpPrimitiveGenerator op_primitive_generator) {

        std::unique_ptr<mkldnn::memory> output_memory_p;
        std::unique_ptr<instant::array> output_arr_p;

        if(required_output_set.find(output_name) != required_output_set.end()) {
            output_arr_p =
              std::make_unique<instant::array>(output_dtype, output_dims);
            output_memory_p = std::make_unique<mkldnn::memory>(
              mkldnn::memory({{{output_dims},
                               dtype_t_to_mkldnn_memory_data_type(output_dtype),
                               output_format},
                              engine},
                             output_arr_p->data()));
        }

        auto op_output_memory =
          output_memory_p ? *output_memory_p : mkldnn::memory(output_pd);
        if(output_memory_p && mkldnn::memory::primitive_desc(output_pd) !=
                                output_memory_p->get_primitive_desc()) {
            op_output_memory = mkldnn::memory(output_pd);
            temp_variable_memory_list.push_back(*output_memory_p);
        }

        net.push_back(op_primitive_generator(op_output_memory));

        if(output_memory_p && op_output_memory != *output_memory_p) {
            net.push_back(mkldnn::reorder(op_output_memory, *output_memory_p));
        }

        variable_memory_list.emplace_back(
          output_name,
          std::make_tuple(std::move(op_output_memory), output_format));
        if(output_arr_p) {
            output_name_and_arr_list.emplace_back(output_name,
                                                  std::move(*output_arr_p));
        }
    }

    inline auto array_to_memory(array const& arr, mkldnn::memory::format format,
                                mkldnn::engine const& engine) {
        return mkldnn::memory({{{arr.dims()},
                                dtype_t_to_mkldnn_memory_data_type(arr.dtype()),
                                format},
                               engine},
                              const_cast<void*>(arr.data()));
    }

    inline auto make_conv_output_dims(mkldnn::memory::dims const& input_tz,
                                      int output_channel_num,
                                      mkldnn::memory::dims const& kernel_shape,
                                      mkldnn::memory::dims const& stride,
                                      mkldnn::memory::dims const& padding_l,
                                      mkldnn::memory::dims const& padding_r) {
        auto calc_length = [](int il, int kl, int pl, int pr, int s) {
            return (il - kl + pl + pr) / s + 1;
        };
        auto batch_size = input_tz[0];
        auto ih = input_tz[2];
        auto iw = input_tz[3];
        auto kh = kernel_shape[0];
        auto kw = kernel_shape[1];
        return mkldnn::memory::dims(
          {batch_size, output_channel_num,
           calc_length(ih, kh, padding_l[0], padding_r[0], stride[0]),
           calc_length(iw, kw, padding_l[1], padding_r[1], stride[1])});
    }

} // namespace instant

#endif // INSTANT_OPERATOR_COMMON_HPP
