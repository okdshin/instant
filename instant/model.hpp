#ifndef INSTANT_MODEL_HPP
#define INSTANT_MODEL_HPP

#include <functional>
#include <iterator>
#include <unordered_map>

#include <instant/array.hpp>
#include <instant/context.hpp>
#include <instant/load_onnx.hpp>

#include <mkldnn.hpp>

namespace instant {
    template <typename Key, typename T>
    inline auto const& find_value(std::unordered_map<Key, T> const& m,
                                  Key const& key) {
        // std::cout << key << std::endl;
        auto found = m.find(key);
        if(found == m.end()) {
            throw std::runtime_error("not found: " + key);
        }
        return found->second;
    }

    template <typename Key, typename T>
    inline auto& find_value(std::unordered_map<Key, T>& m, Key const& key) {
        // std::cout << key << std::endl;
        auto found = m.find(key);
        if(found == m.end()) {
            throw std::runtime_error("not found: " + key);
        }
        return found->second;
    }

    inline auto extract_dims(mkldnn::memory const& m) {
        auto const& d = m.get_primitive_desc().desc().data;
        return std::vector<int>(d.dims, d.dims + d.ndims);
    }

    inline auto make_conv_output_tz(mkldnn::memory::dims const& input_tz,
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

    inline auto make_conv_primitive(
      std::unordered_map<std::string, const mkldnn::memory> const&
        parameter_memory_table,
      std::unordered_map<std::string, std::tuple<const mkldnn::memory,
                                                 mkldnn::memory::format>> const&
        variable_memory_table,
      std::set<std::string> const& required_output_set,
      onnx::NodeProto const& node, mkldnn::engine const& engine) {

        // Load attributes
        using namespace std::literals::string_literals;

        auto attribute_table = instant::make_attribute_table(node);

        onnx::AttributeProto const& strides_attr =
          find_value(attribute_table, "strides"s);
        assert(strides_attr.ints().size() == 2);
        auto strides = mkldnn::memory::dims(strides_attr.ints().begin(),
                                            strides_attr.ints().end());

        onnx::AttributeProto const& kernel_shape_attr =
          find_value(attribute_table, "kernel_shape"s);
        assert(kernel_shape_attr.ints().size() == 2);
        auto kernel_shape = mkldnn::memory::dims(
          kernel_shape_attr.ints().begin(), kernel_shape_attr.ints().end());

        onnx::AttributeProto const& pads_attr =
          find_value(attribute_table, "pads"s);
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
            throw std::runtime_error("Not implemented"); // TODO
        }

        // Load input and weight
        auto const& input_memory_and_origin_format =
          find_value(variable_memory_table, node.input(0));
        auto const& input_memory = std::get<0>(input_memory_and_origin_format);
        auto input_origin_format = std::get<1>(input_memory_and_origin_format);
        auto const& weight_memory =
          find_value(parameter_memory_table, node.input(1));
        auto input_dims = extract_dims(input_memory);
        auto weight_dims = extract_dims(weight_memory);
        auto output_tz =
          make_conv_output_tz(input_dims, weight_dims[0], kernel_shape, strides,
                              padding_l, padding_r);
        std::unique_ptr<mkldnn::memory> output_memory_p;
        std::unique_ptr<instant::array> output_arr_p;
        auto const& output_name = node.output(0);
        std::vector<mkldnn::memory>
          temp_variable_memory_list; // for temporary memory's life
        if(required_output_set.find(output_name) != required_output_set.end()) {
            output_arr_p =
              std::make_unique<instant::array>(dtype_t::float_, output_tz);
            output_memory_p = std::make_unique<mkldnn::memory>(
              mkldnn::memory({{{output_tz},
                               mkldnn::memory::data_type::f32,
                               input_origin_format},
                              engine},
                             output_arr_p->data()));
        }

        auto conv_input_md =
          mkldnn::memory::desc({input_dims}, mkldnn::memory::data_type::f32,
                               mkldnn::memory::format::any);
        auto conv_weight_md =
          mkldnn::memory::desc({weight_dims}, mkldnn::memory::data_type::f32,
                               mkldnn::memory::format::any);
        auto conv_output_md =
          mkldnn::memory::desc({output_tz}, mkldnn::memory::data_type::f32,
                               mkldnn::memory::format::any);

        std::unique_ptr<mkldnn::convolution_forward::desc> conv_desc_p;
        if(node.input_size() == 2) {
            conv_desc_p = std::make_unique<mkldnn::convolution_forward::desc>(
              mkldnn::prop_kind::forward_inference, mkldnn::algorithm::convolution_direct,
              conv_input_md, conv_weight_md, conv_output_md, strides, padding_l,
              padding_r, mkldnn::padding_kind::zero);
        } else {
            auto const& bias_memory =
              find_value(parameter_memory_table, node.input(2));
            conv_desc_p = std::make_unique<mkldnn::convolution_forward::desc>(
              mkldnn::prop_kind::forward_inference, mkldnn::algorithm::convolution_direct,
              conv_input_md, conv_weight_md,
              bias_memory.get_primitive_desc().desc(), conv_output_md, strides,
              padding_l, padding_r, mkldnn::padding_kind::zero);
        }
        auto conv_pd =
          mkldnn::convolution_forward::primitive_desc(*conv_desc_p, engine);

        std::vector<mkldnn::primitive> net;

        auto conv_input_memory = input_memory;
        if(mkldnn::memory::primitive_desc(conv_pd.src_primitive_desc()) !=
           input_memory.get_primitive_desc()) {
            conv_input_memory = mkldnn::memory(conv_pd.src_primitive_desc());
            temp_variable_memory_list.push_back(conv_input_memory);
            net.push_back(mkldnn::reorder(input_memory, conv_input_memory));
        }

        auto conv_weight_memory = weight_memory;
        if(mkldnn::memory::primitive_desc(conv_pd.weights_primitive_desc()) !=
           weight_memory.get_primitive_desc()) {
            conv_weight_memory =
              mkldnn::memory(conv_pd.weights_primitive_desc());
            temp_variable_memory_list.push_back(conv_weight_memory);
            net.push_back(mkldnn::reorder(weight_memory, conv_weight_memory));
        }

        auto conv_output_memory =
          output_memory_p ? *output_memory_p
                          : mkldnn::memory(conv_pd.dst_primitive_desc());
        if(output_memory_p &&
           mkldnn::memory::primitive_desc(conv_pd.dst_primitive_desc()) !=
             output_memory_p->get_primitive_desc()) {
            conv_output_memory = mkldnn::memory(conv_pd.dst_primitive_desc());
            temp_variable_memory_list.push_back(*output_memory_p);
        }

        if(node.input_size() == 2) {
            net.push_back(mkldnn::convolution_forward(
              conv_pd, conv_input_memory, conv_weight_memory,
              conv_output_memory));
        } else {
            auto const& conv_bias_memory =
              find_value(parameter_memory_table, node.input(2));
            net.push_back(mkldnn::convolution_forward(
              conv_pd, conv_input_memory, conv_weight_memory, conv_bias_memory,
              conv_output_memory));
        }

        if(output_memory_p && conv_output_memory != *output_memory_p) {
            net.push_back(
              mkldnn::reorder(conv_output_memory, *output_memory_p));
        }

        auto output_name_and_mem_and_origin_format = std::make_pair(
          output_name,
          std::make_tuple(std::move(conv_output_memory), input_origin_format));
        std::vector<std::pair<std::string, array>> output_name_and_arr_list;
        if(output_arr_p) {
            output_name_and_arr_list.emplace_back(
              std::make_pair(output_name, std::move(*output_arr_p)));
        }
        return std::make_tuple(
          net,
          std::vector<decltype(output_name_and_mem_and_origin_format)>{
            std::move(output_name_and_mem_and_origin_format)},
          temp_variable_memory_list, output_name_and_arr_list);
    }

    inline auto make_relu_primitive(
      std::unordered_map<std::string, const mkldnn::memory> const&
      /*parameter_memory_table*/,
      std::unordered_map<std::string, std::tuple<const mkldnn::memory,
                                                 mkldnn::memory::format>> const&
        variable_memory_table,
      std::set<std::string> const& required_output_set,
      onnx::NodeProto const& node, mkldnn::engine const& engine) {
        auto negative_slope = 0.; // 1.0;
        auto const& input_memory_and_origin_format =
          find_value(variable_memory_table, node.input(0));
        auto const& input_memory = std::get<0>(input_memory_and_origin_format);
        auto input_origin_format = std::get<1>(input_memory_and_origin_format);
        auto input_output_dims = extract_dims(input_memory);

        std::unique_ptr<mkldnn::memory> output_memory_p;
        std::unique_ptr<instant::array> output_arr_p;
        auto const& output_name = node.output(0);
        std::vector<mkldnn::memory>
          temp_variable_memory_list; // for temporary memory's life
        if(required_output_set.find(output_name) != required_output_set.end()) {
            output_arr_p = std::make_unique<instant::array>(dtype_t::float_,
                                                            input_output_dims);
            output_memory_p = std::make_unique<mkldnn::memory>(
              mkldnn::memory({{{input_output_dims},
                               mkldnn::memory::data_type::f32,
                               input_origin_format},
                              engine},
                             output_arr_p->data()));
        }

        auto op_desc = mkldnn::eltwise_forward::desc(
          mkldnn::prop_kind::forward_inference, mkldnn::algorithm::eltwise_relu,
          input_memory.get_primitive_desc().desc(), negative_slope);
        auto op_pd = mkldnn::eltwise_forward::primitive_desc(op_desc, engine);

        std::vector<mkldnn::primitive> net;

        auto op_output_memory = output_memory_p
                                  ? *output_memory_p
                                  : mkldnn::memory(op_pd.dst_primitive_desc());
        if(output_memory_p &&
           mkldnn::memory::primitive_desc(op_pd.dst_primitive_desc()) !=
             output_memory_p->get_primitive_desc()) {
            op_output_memory = mkldnn::memory(op_pd.dst_primitive_desc());
            temp_variable_memory_list.push_back(*output_memory_p);
        }

        net.push_back(
          mkldnn::eltwise_forward(op_pd, input_memory, op_output_memory));

        if(output_memory_p && op_output_memory != *output_memory_p) {
            std::cout << "reorder" << std::endl;
            net.push_back(mkldnn::reorder(op_output_memory, *output_memory_p));
        }

        auto output_name_and_mem_and_origin_format = std::make_pair(
          output_name,
          std::make_tuple(std::move(op_output_memory), input_origin_format));
        std::vector<std::pair<std::string, array>> output_name_and_arr_list;
        if(output_arr_p) {
            output_name_and_arr_list.emplace_back(
              std::make_pair(output_name, std::move(*output_arr_p)));
        }
        return std::make_tuple(
          net,
          std::vector<decltype(output_name_and_mem_and_origin_format)>{
            std::move(output_name_and_mem_and_origin_format)},
          temp_variable_memory_list, output_name_and_arr_list);
    }

    inline auto make_max_pool_primitive(
      std::unordered_map<std::string, const mkldnn::memory> const&
      /*parameter_memory_table*/,
      std::unordered_map<std::string, std::tuple<const mkldnn::memory,
                                                 mkldnn::memory::format>> const&
        variable_memory_table,
      std::set<std::string> const& required_output_set,
      onnx::NodeProto const& node, mkldnn::engine const& engine) {
        // Load attributes
        using namespace std::literals::string_literals;

        auto attribute_table = instant::make_attribute_table(node);

        onnx::AttributeProto const& strides_attr =
          find_value(attribute_table, "strides"s);
        assert(strides_attr.ints().size() == 2);
        auto strides = mkldnn::memory::dims(strides_attr.ints().begin(),
                                            strides_attr.ints().end());

        onnx::AttributeProto const& kernel_shape_attr =
          find_value(attribute_table, "kernel_shape"s);
        assert(kernel_shape_attr.ints().size() == 2);
        auto kernel_shape = mkldnn::memory::dims(
          kernel_shape_attr.ints().begin(), kernel_shape_attr.ints().end());

        onnx::AttributeProto const& pads_attr =
          find_value(attribute_table, "pads"s);
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
            throw std::runtime_error("Not implemented"); // TODO
        }

        // Load input and weight
        auto const& input_memory_and_origin_format =
          find_value(variable_memory_table, node.input(0));
        auto const& input_memory = std::get<0>(input_memory_and_origin_format);
        auto input_origin_format = std::get<1>(input_memory_and_origin_format);
        auto input_dims = extract_dims(input_memory);
        auto output_channel_num = input_dims[1];
        auto output_tz =
          make_conv_output_tz(input_dims, output_channel_num, kernel_shape,
                              strides, padding_l, padding_r);
        std::unique_ptr<mkldnn::memory> output_memory_p;
        std::unique_ptr<instant::array> output_arr_p;
        auto const& output_name = node.output(0);
        std::vector<mkldnn::memory>
          temp_variable_memory_list; // for temporary memory's life
        if(required_output_set.find(output_name) != required_output_set.end()) {
            output_arr_p =
              std::make_unique<instant::array>(dtype_t::float_, output_tz);
            output_memory_p = std::make_unique<mkldnn::memory>(
              mkldnn::memory({{{output_tz},
                               mkldnn::memory::data_type::f32,
                               input_origin_format},
                              engine},
                             output_arr_p->data()));
        }

        auto max_pool_output_md =
          mkldnn::memory::desc({output_tz}, mkldnn::memory::data_type::f32,
                               mkldnn::memory::format::any);

        auto max_pool_desc = mkldnn::pooling_forward::desc(
          mkldnn::prop_kind::forward, mkldnn::pooling_max,
          input_memory.get_primitive_desc().desc(), max_pool_output_md, strides,
          kernel_shape, padding_l, padding_r, mkldnn::padding_kind::zero);
        auto max_pool_pd =
          mkldnn::pooling_forward::primitive_desc(max_pool_desc, engine);

        std::vector<mkldnn::primitive> net;

        auto max_pool_output_memory =
          output_memory_p ? *output_memory_p
                          : mkldnn::memory(max_pool_pd.dst_primitive_desc());
        if(output_memory_p &&
           mkldnn::memory::primitive_desc(max_pool_pd.dst_primitive_desc()) !=
             output_memory_p->get_primitive_desc()) {
            max_pool_output_memory =
              mkldnn::memory(max_pool_pd.dst_primitive_desc());
            temp_variable_memory_list.push_back(*output_memory_p);
        }

        auto max_pool_indices_memory =
          mkldnn::memory(max_pool_pd.workspace_primitive_desc());
        temp_variable_memory_list.push_back(max_pool_indices_memory);

        net.push_back(mkldnn::pooling_forward(max_pool_pd, input_memory,
                                              max_pool_output_memory,
                                              max_pool_indices_memory));

        if(output_memory_p && max_pool_output_memory != *output_memory_p) {
            std::cout << "reorder" << std::endl;
            net.push_back(
              mkldnn::reorder(max_pool_output_memory, *output_memory_p));
        }

        auto output_name_and_mem_and_origin_format = std::make_pair(
          output_name, std::make_tuple(std::move(max_pool_output_memory),
                                       input_origin_format));
        std::vector<std::pair<std::string, array>> output_name_and_arr_list;
        if(output_arr_p) {
            output_name_and_arr_list.emplace_back(
              std::make_pair(output_name, std::move(*output_arr_p)));
        }

        return std::make_tuple(
          net,
          std::vector<decltype(output_name_and_mem_and_origin_format)>{
            std::move(output_name_and_mem_and_origin_format)},
          temp_variable_memory_list, output_name_and_arr_list);
    }

    inline auto make_nop_primitive(
      std::unordered_map<std::string, const mkldnn::memory> const&
      /*parameter_memory_table*/,
      std::unordered_map<std::string, std::tuple<const mkldnn::memory,
                                                 mkldnn::memory::format>> const&
        variable_memory_table,
      std::set<std::string> const& required_output_set,
      onnx::NodeProto const& node, mkldnn::engine const& engine) {
        auto const& input_memory_and_origin_format =
          find_value(variable_memory_table, node.input(0));
        auto const& input_memory = std::get<0>(input_memory_and_origin_format);
        auto input_origin_format = std::get<1>(input_memory_and_origin_format);
        auto input_output_dims = extract_dims(input_memory);

        std::unique_ptr<mkldnn::memory> output_memory_p;
        std::unique_ptr<instant::array> output_arr_p;
        auto const& output_name = node.output(0);
        std::vector<mkldnn::memory>
          temp_variable_memory_list; // for temporary memory's life
        if(required_output_set.find(output_name) != required_output_set.end()) {
            output_arr_p = std::make_unique<instant::array>(dtype_t::float_,
                                                            input_output_dims);
            output_memory_p = std::make_unique<mkldnn::memory>(
              mkldnn::memory({{{input_output_dims},
                               mkldnn::memory::data_type::f32,
                               input_origin_format},
                              engine},
                             output_arr_p->data()));
        }

        std::vector<mkldnn::primitive> net;

        auto op_output_memory =
          output_memory_p ? *output_memory_p
                          : mkldnn::memory(input_memory.get_primitive_desc());
        if(output_memory_p &&
           mkldnn::memory::primitive_desc(input_memory.get_primitive_desc()) !=
             output_memory_p->get_primitive_desc()) {
            op_output_memory =
              mkldnn::memory(input_memory.get_primitive_desc());
            temp_variable_memory_list.push_back(*output_memory_p);
        }

        net.push_back(mkldnn::reorder(input_memory, op_output_memory));

        if(output_memory_p && op_output_memory != *output_memory_p) {
            std::cout << "reorder" << std::endl;
            net.push_back(mkldnn::reorder(op_output_memory, *output_memory_p));
        }

        auto output_name_and_mem_and_origin_format = std::make_pair(
          output_name,
          std::make_tuple(std::move(op_output_memory), input_origin_format));
        std::vector<std::pair<std::string, array>> output_name_and_arr_list;
        if(output_arr_p) {
            output_name_and_arr_list.emplace_back(
              std::make_pair(output_name, std::move(*output_arr_p)));
        }
        return std::make_tuple(
          net,
          std::vector<decltype(output_name_and_mem_and_origin_format)>{
            std::move(output_name_and_mem_and_origin_format)},
          temp_variable_memory_list, output_name_and_arr_list);
    }

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

        // Load attributes
        using namespace std::literals::string_literals;

        onnx::AttributeProto const& shape_attr =
          find_value(attribute_table, "shape"s);
        auto shape = mkldnn::memory::dims(shape_attr.ints().begin(),
                                          shape_attr.ints().end());
        for(auto s : shape) {
            std::cout << s << " ";
        }
        std::cout << "\n";

        auto const& input_memory_and_origin_format =
          find_value(variable_memory_table, node.input(0));
        auto const& input_memory = std::get<0>(input_memory_and_origin_format);
        /*
        auto input_origin_format =
          std::get<1>(input_memory_and_origin_format);
        */
        auto input_dims = extract_dims(input_memory);
        for(auto s : input_dims) {
            std::cout << s << " ";
        }
        std::cout << "\n";
        auto output_dims = calc_reshaped_dims(input_dims, shape);
        for(auto s : output_dims) {
            std::cout << s << " ";
        }
        std::cout << "\n";

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

    inline auto make_fc_primitive(
      std::unordered_map<std::string, const mkldnn::memory> const&
        parameter_memory_table,
      std::unordered_map<std::string, std::tuple<const mkldnn::memory,
                                                 mkldnn::memory::format>> const&
        variable_memory_table,
      std::set<std::string> const& required_output_set,
      onnx::NodeProto const& node, mkldnn::engine const& engine) {
        // Load attributes
        using namespace std::literals::string_literals;

        auto attribute_table = instant::make_attribute_table(node);

        onnx::AttributeProto const& axis_attr =
          find_value(attribute_table, "axis"s);
        auto axis = axis_attr.i();
        assert(axis == 1); // TODO

        onnx::AttributeProto const& axis_w_attr =
          find_value(attribute_table, "axis_w"s);
        auto axis_w = axis_w_attr.i();
        assert(axis_w == 1); // TODO

        // Load input and weight
        auto const& input_memory_and_origin_format =
          find_value(variable_memory_table, node.input(0));
        auto const& input_memory = std::get<0>(input_memory_and_origin_format);
        auto input_origin_format = std::get<1>(input_memory_and_origin_format);
        auto const& weight_memory =
          find_value(parameter_memory_table, node.input(1));
        auto const& bias_memory =
          find_value(parameter_memory_table, node.input(2));
        auto input_dims = extract_dims(input_memory);
        auto weight_dims = extract_dims(weight_memory);
        auto bias_dims = extract_dims(bias_memory);
        mkldnn::memory::dims output_tz{input_dims[0], bias_dims[0]};

        std::unique_ptr<mkldnn::memory> output_memory_p;
        std::unique_ptr<instant::array> output_arr_p;
        auto const& output_name = node.output(0);
        std::vector<mkldnn::memory>
          temp_variable_memory_list; // for temporary memory's life
        if(required_output_set.find(output_name) != required_output_set.end()) {
            output_arr_p =
              std::make_unique<instant::array>(dtype_t::float_, output_tz);
            output_memory_p = std::make_unique<mkldnn::memory>(
              mkldnn::memory({{{output_tz},
                               mkldnn::memory::data_type::f32,
                               mkldnn::memory::format::nc},
                              engine},
                             output_arr_p->data()));
        }
        for(auto i : input_dims) {
            std::cout << i << " ";
        }
        std::cout << "\n";
        for(auto i : weight_dims) {
            std::cout << i << " ";
        }
        std::cout << "\n";
        for(auto i : output_tz) {
            std::cout << i << " ";
        }
        std::cout << "\n";

        auto fc_input_md =
          mkldnn::memory::desc({input_dims}, mkldnn::memory::data_type::f32,
                               mkldnn::memory::format::any);
        auto fc_weight_md =
          mkldnn::memory::desc({weight_dims}, mkldnn::memory::data_type::f32,
                               mkldnn::memory::format::any);
        auto fc_output_md =
          mkldnn::memory::desc({output_tz}, mkldnn::memory::data_type::f32,
                               mkldnn::memory::format::any);

        mkldnn::inner_product_forward::desc fc_desc(
          mkldnn::prop_kind::forward_inference, fc_input_md, fc_weight_md,
          bias_memory.get_primitive_desc().desc(), fc_output_md);
        auto fc_pd =
          mkldnn::inner_product_forward::primitive_desc(fc_desc, engine);

        std::vector<mkldnn::primitive> net;

        auto fc_input_memory = input_memory;
        if(mkldnn::memory::primitive_desc(fc_pd.src_primitive_desc()) !=
           input_memory.get_primitive_desc()) {
            fc_input_memory = mkldnn::memory(fc_pd.src_primitive_desc());
            temp_variable_memory_list.push_back(fc_input_memory);
            net.push_back(mkldnn::reorder(input_memory, fc_input_memory));
        }

        auto fc_weight_memory = weight_memory;
        if(mkldnn::memory::primitive_desc(fc_pd.weights_primitive_desc()) !=
           weight_memory.get_primitive_desc()) {
            fc_weight_memory = mkldnn::memory(fc_pd.weights_primitive_desc());
            temp_variable_memory_list.push_back(fc_weight_memory);
            net.push_back(mkldnn::reorder(weight_memory, fc_weight_memory));
        }

        auto fc_output_memory = output_memory_p
                                  ? *output_memory_p
                                  : mkldnn::memory(fc_pd.dst_primitive_desc());
        if(output_memory_p &&
           mkldnn::memory::primitive_desc(fc_pd.dst_primitive_desc()) !=
             output_memory_p->get_primitive_desc()) {
            fc_output_memory = mkldnn::memory(fc_pd.dst_primitive_desc());
            temp_variable_memory_list.push_back(*output_memory_p);
        }

        net.push_back(mkldnn::inner_product_forward(
          fc_pd, fc_input_memory, fc_weight_memory, bias_memory,
          fc_output_memory));

        if(output_memory_p && fc_output_memory != *output_memory_p) {
            net.push_back(mkldnn::reorder(fc_output_memory, *output_memory_p));
        }

        auto output_name_and_mem_and_origin_format = std::make_pair(
          output_name,
          std::make_tuple(std::move(fc_output_memory), input_origin_format));
        std::vector<std::pair<std::string, array>> output_name_and_arr_list;
        if(output_arr_p) {
            output_name_and_arr_list.emplace_back(
              std::make_pair(output_name, std::move(*output_arr_p)));
        }
        return std::make_tuple(
          net,
          std::vector<decltype(output_name_and_mem_and_origin_format)>{
            std::move(output_name_and_mem_and_origin_format)},
          temp_variable_memory_list, output_name_and_arr_list);
    }

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

    inline auto make_softmax_primitive(
      std::unordered_map<std::string, const mkldnn::memory> const&
      /*parameter_memory_table*/,
      std::unordered_map<std::string, std::tuple<const mkldnn::memory,
                                                 mkldnn::memory::format>> const&
        variable_memory_table,
      std::set<std::string> const& required_output_set,
      onnx::NodeProto const& node, mkldnn::engine const& engine) {
        constexpr auto softmax_axis = 1;
        auto const& input_memory_and_origin_format =
          find_value(variable_memory_table, node.input(0));
        auto const& input_memory = std::get<0>(input_memory_and_origin_format);
        auto input_origin_format = std::get<1>(input_memory_and_origin_format);
        auto input_output_dims = extract_dims(input_memory);

        std::unique_ptr<mkldnn::memory> output_memory_p;
        std::unique_ptr<instant::array> output_arr_p;
        auto const& output_name = node.output(0);
        std::vector<mkldnn::memory>
          temp_variable_memory_list; // for temporary memory's life
        if(required_output_set.find(output_name) != required_output_set.end()) {
            output_arr_p = std::make_unique<instant::array>(dtype_t::float_,
                                                            input_output_dims);
            output_memory_p = std::make_unique<mkldnn::memory>(
              mkldnn::memory({{{input_output_dims},
                               mkldnn::memory::data_type::f32,
                               input_origin_format},
                              engine},
                             output_arr_p->data()));
        }

        auto op_desc = mkldnn::softmax_forward::desc(
          mkldnn::prop_kind::forward_inference, input_memory.get_primitive_desc().desc(),
          softmax_axis);
        auto op_pd = mkldnn::softmax_forward::primitive_desc(op_desc, engine);

        std::vector<mkldnn::primitive> net;

        auto op_output_memory =
          output_memory_p ? *output_memory_p
                          : mkldnn::memory(input_memory.get_primitive_desc());

        net.push_back(
          mkldnn::softmax_forward(op_pd, input_memory, op_output_memory));

        auto output_name_and_mem_and_origin_format = std::make_pair(
          output_name,
          std::make_tuple(std::move(op_output_memory), input_origin_format));
        std::vector<std::pair<std::string, array>> output_name_and_arr_list;
        if(output_arr_p) {
            output_name_and_arr_list.emplace_back(
              std::make_pair(output_name, std::move(*output_arr_p)));
        }
        return std::make_tuple(
          net,
          std::vector<decltype(output_name_and_mem_and_origin_format)>{
            std::move(output_name_and_mem_and_origin_format)},
          temp_variable_memory_list, output_name_and_arr_list);
    }

    inline auto make_parameter_memory_pair(
      onnx::NodeProto const& node, int param_index,
      mkldnn::memory::format format,
      std::unordered_map<std::string, instant::array>& parameter_table,
      mkldnn::engine const& engine) {
        auto const& name = node.input(param_index);
        auto& arr = find_value(parameter_table, name);
        mkldnn::memory::dims tz(arr.dims().begin(), arr.dims().end());
        auto mem = mkldnn::memory(
          {{{tz}, mkldnn::memory::data_type::f32, format}, engine}, arr.data());
        return std::make_pair(name, mem);
    }

    inline auto make_parameter_memory_table(
      onnx::GraphProto const& graph,
      std::unordered_map<std::string, instant::array>& parameter_table,
      mkldnn::engine const& engine) {
        std::unordered_map<std::string, const mkldnn::memory> memory_table;
        for(auto const& node : graph.node()) {
            if(node.op_type() == "Conv") {
                constexpr auto weight_index = 1;
                memory_table.insert(make_parameter_memory_pair(
                  node, weight_index, mkldnn::memory::format::oihw,
                  parameter_table, engine));

                if(node.input_size() != 2) {
                    constexpr auto bias_index = 2;
                    memory_table.insert(make_parameter_memory_pair(
                      node, bias_index, mkldnn::memory::format::x,
                      parameter_table, engine));
                }
            } else if(node.op_type() == "FC") {
                constexpr auto weight_index = 1;
                constexpr auto bias_index = 2;
                memory_table.insert(make_parameter_memory_pair(
                  node, weight_index, mkldnn::memory::format::oi, // TODO check
                  parameter_table, engine));
                memory_table.insert(make_parameter_memory_pair(
                  node, bias_index, mkldnn::memory::format::x, parameter_table,
                  engine));
            } else {
                // TODO
                // throw std::runtime_error("Not implemented: " +
                // node.op_type());
            }
        }
        return memory_table;
    }

    inline auto make_variable_memory_table(
      std::vector<std::tuple<std::string, instant::array,
                             mkldnn::memory::format>>& input_list,
      mkldnn::engine const& engine) {
        std::unordered_map<
          std::string, std::tuple<const mkldnn::memory, mkldnn::memory::format>>
          memory_table;
        for(auto& input : input_list) {
            auto& arr = std::get<1>(input);
            auto format = std::get<2>(input);
            mkldnn::memory::dims tz(arr.dims().begin(), arr.dims().end());
            auto mem = mkldnn::memory(
              {{{tz}, mkldnn::memory::data_type::f32, format}, engine},
              arr.data());
            auto const& name = std::get<0>(input);
            memory_table.insert({name, {mem, format}});
        }
        return memory_table;
    }

    using primitive_factory =
      std::function<
        std::tuple<
          std::vector<mkldnn::primitive>, // net
          std::vector<std::pair<
            std::string, std::tuple<mkldnn::memory,
                                    mkldnn::memory::format>>>, // output name
                                                               // and [memory
                                                               // and origin
                                                               // format] list
          std::vector<mkldnn::memory>, // temporary variable memory list
          std::vector<std::pair<std::string, array>>> // reqired output
                                                      // name and array
                                                      // list
        (std::unordered_map<
           std::string, const mkldnn::memory> const&, // parameter memory table
         std::unordered_map<
           std::string,
           std::tuple<const mkldnn::memory,
                      mkldnn::memory::format>> const&, // variable memory
                                                       // table
         std::set<std::string> const&, // required output name set
         onnx::NodeProto const&, mkldnn::engine const&)>;

    inline auto make_default_primitive_factory_table() {
        std::unordered_map<std::string, primitive_factory>
          primitive_factory_table;
        primitive_factory_table.insert({"Conv", make_conv_primitive});
        primitive_factory_table.insert({"Relu", make_relu_primitive});
        primitive_factory_table.insert({"MaxPool", make_max_pool_primitive});
        primitive_factory_table.insert({"Reshape", make_reshape_primitive});
        // primitive_factory_table.insert({"Reshape", make_nop_primitive});
        primitive_factory_table.insert({"FC", make_fc_primitive});
        primitive_factory_table.insert({"Dropout", make_dropout_primitive});
        primitive_factory_table.insert({"Softmax", make_softmax_primitive});
        // TODO other primitives
        return primitive_factory_table;
    }

    inline auto run_model(
      onnx::GraphProto const& graph,
      std::unordered_map<std::string, const mkldnn::memory> const&
        parameter_memory_table,
      std::unordered_map<
        std::string, std::tuple<const mkldnn::memory, mkldnn::memory::format>>&
        variable_memory_table,
      std::set<std::string> const& required_output_set,
      std::unordered_map<std::string, primitive_factory>
        primitive_factory_table =
          instant::make_default_primitive_factory_table(),
      instant::context const& context = instant::get_context()) {
        std::unordered_map<std::string, instant::array> output_table;
        std::vector<mkldnn::primitive> nets;
        std::vector<mkldnn::memory> temp_variable_memory_list;
        for(auto const& node : graph.node()) {
            std::cout << node.op_type() << "\t";
            for(auto const& i : node.input()) {
                std::cout << i << " ";
            }
            std::cout << "-> ";
            for(auto const& i : node.output()) {
                std::cout << i << " ";
            }
            std::cout << "\n";
            try {
                auto primitive_factory_pair_iter =
                  primitive_factory_table.find(node.op_type());
                if(primitive_factory_pair_iter ==
                   primitive_factory_table.end()) {
                    throw std::runtime_error("Implementation not found: " +
                                             node.op_type());
                }
                auto temp_tuple =
                  primitive_factory_pair_iter->second.operator()(
                    parameter_memory_table, variable_memory_table,
                    required_output_set, node, context.engine());
                auto& net = std::get<0>(temp_tuple);
                auto& output_name_and_memory_and_origin_format_list =
                  std::get<1>(temp_tuple);
                auto& temp_vars = std::get<2>(temp_tuple);
                auto& output_name_and_arr_list = std::get<3>(temp_tuple);

                nets.insert(nets.end(), std::make_move_iterator(net.begin()),
                            std::make_move_iterator(net.end()));
                variable_memory_table.insert(
                  std::make_move_iterator(
                    output_name_and_memory_and_origin_format_list.begin()),
                  std::make_move_iterator(
                    output_name_and_memory_and_origin_format_list.end()));
                temp_variable_memory_list.insert(
                  temp_variable_memory_list.end(),
                  std::make_move_iterator(temp_vars.begin()),
                  std::make_move_iterator(temp_vars.end()));
                output_table.insert(
                  std::make_move_iterator(output_name_and_arr_list.begin()),
                  std::make_move_iterator(output_name_and_arr_list.end()));
            } catch(mkldnn::error const& e) {
                std::cout << "MKLDNN Error: " << e.message << std::endl;
            } catch(std::exception const& e) {
                std::cout << "Error: " << e.what() << std::endl;
            }
        }
        std::cout << "float size " << sizeof(float) << std::endl;
        std::cout << "net size " << nets.size() << std::endl;
        mkldnn::stream(mkldnn::stream::kind::eager).submit(nets).wait();
        return output_table;
    }

} // namespace instant
#endif // INSTANT_MODEL_HPP
