#ifndef INSTANT_OPERATOR_HPP
#define INSTANT_OPERATOR_HPP

#include <unordered_map>

#include <mkldnn.hpp>

#include <instant/array.hpp>
#include <instant/context.hpp>
#include <instant/load_onnx.hpp>

namespace instant {

    template <typename T>
    auto const& find_value(std::unordered_map<std::string, T> const& m,
                           std::string const& key) {
        // std::cout << key << std::endl;
        auto found = m.find(key);
        if(found == m.end()) {
            throw std::runtime_error("not found: " + key);
        }
        return found->second;
    }

    template <typename T>
    auto& find_value(std::unordered_map<std::string, T>& m,
                     std::string const& key) {
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
            throw std::runtime_error("Not implemented"); // TODO
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

    inline auto make_conv_primitive(
      std::unordered_map<std::string, const mkldnn::memory> const&
        parameter_memory_table,
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
        auto const& weight_memory =
          find_value(parameter_memory_table, node.input(1));

        auto input_dims = extract_dims(input_memory);
        auto weight_dims = extract_dims(weight_memory);
        auto output_dims =
          make_conv_output_dims(input_dims, weight_dims[0], kernel_shape,
                                strides, padding_l, padding_r);

        auto const& output_name = node.output(0);

        auto conv_input_md =
          mkldnn::memory::desc({input_dims}, mkldnn::memory::data_type::f32,
                               mkldnn::memory::format::any);
        auto conv_weight_md =
          mkldnn::memory::desc({weight_dims}, mkldnn::memory::data_type::f32,
                               mkldnn::memory::format::any);
        auto conv_output_md =
          mkldnn::memory::desc({output_dims}, mkldnn::memory::data_type::f32,
                               mkldnn::memory::format::any);

        std::unique_ptr<mkldnn::convolution_forward::desc> conv_desc_p;
        if(node.input_size() == 2) {
            conv_desc_p = std::make_unique<mkldnn::convolution_forward::desc>(
              mkldnn::prop_kind::forward_inference,
              mkldnn::algorithm::convolution_direct, conv_input_md,
              conv_weight_md, conv_output_md, strides, padding_l, padding_r,
              mkldnn::padding_kind::zero);
        } else {
            auto const& bias_memory =
              find_value(parameter_memory_table, node.input(2));
            conv_desc_p = std::make_unique<mkldnn::convolution_forward::desc>(
              mkldnn::prop_kind::forward_inference,
              mkldnn::algorithm::convolution_direct, conv_input_md,
              conv_weight_md, bias_memory.get_primitive_desc().desc(),
              conv_output_md, strides, padding_l, padding_r,
              mkldnn::padding_kind::zero);
        }
        auto conv_pd =
          mkldnn::convolution_forward::primitive_desc(*conv_desc_p, engine);

        std::vector<mkldnn::primitive> net;
        std::vector<mkldnn::memory>
          temp_variable_memory_list; // for temporary memory's life

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

        std::vector<std::pair<
          std::string, std::tuple<mkldnn::memory, mkldnn::memory::format>>>
          variable_memory_list;
        std::vector<std::pair<std::string, array>> output_name_and_arr_list;

        manage_output_memory(
          required_output_set, output_name, dtype_t::float_, output_dims,
          input_origin_format, conv_pd.dst_primitive_desc(),
          variable_memory_list, temp_variable_memory_list,
          output_name_and_arr_list, net, engine,
          [&conv_input_memory, &conv_weight_memory, &node, &conv_pd,
           &parameter_memory_table](auto& op_output_memory) {
              if(node.input_size() == 2) {
                  return mkldnn::convolution_forward(conv_pd, conv_input_memory,
                                                     conv_weight_memory,
                                                     op_output_memory);
              } else {
                  auto const& conv_bias_memory =
                    find_value(parameter_memory_table, node.input(2));
                  return mkldnn::convolution_forward(
                    conv_pd, conv_input_memory, conv_weight_memory,
                    conv_bias_memory, op_output_memory);
              }
          });

        return std::make_tuple(net, variable_memory_list,
                               temp_variable_memory_list,
                               output_name_and_arr_list);
    }

    inline auto make_relu_primitive(
      std::unordered_map<std::string, const mkldnn::memory> const&
      /*parameter_memory_table*/,
      std::unordered_map<std::string, std::tuple<const mkldnn::memory,
                                                 mkldnn::memory::format>> const&
        variable_memory_table,
      std::set<std::string> const& required_output_set,
      onnx::NodeProto const& node, mkldnn::engine const& engine) {
        auto negative_slope = 0.;
        auto const& input_memory_and_origin_format =
          find_value(variable_memory_table, node.input(0));
        auto const& input_memory = std::get<0>(input_memory_and_origin_format);
        auto input_origin_format = std::get<1>(input_memory_and_origin_format);
        auto input_output_dims = extract_dims(input_memory);

        auto const& output_name = node.output(0);

        auto op_desc = mkldnn::eltwise_forward::desc(
          mkldnn::prop_kind::forward_inference, mkldnn::algorithm::eltwise_relu,
          input_memory.get_primitive_desc().desc(), negative_slope);
        auto op_pd = mkldnn::eltwise_forward::primitive_desc(op_desc, engine);

        std::vector<mkldnn::primitive> net;
        std::vector<std::pair<
          std::string, std::tuple<mkldnn::memory, mkldnn::memory::format>>>
          variable_memory_list;
        std::vector<mkldnn::memory>
          temp_variable_memory_list; // for temporary memory's life
        std::vector<std::pair<std::string, array>> output_name_and_arr_list;

        manage_output_memory(
          required_output_set, output_name, dtype_t::float_, input_output_dims,
          input_origin_format, op_pd.dst_primitive_desc(), variable_memory_list,
          temp_variable_memory_list, output_name_and_arr_list, net, engine,
          [&input_memory, &node, &op_pd](auto& op_output_memory) {
              return mkldnn::eltwise_forward(op_pd, input_memory,
                                             op_output_memory);
          });

        return std::make_tuple(net, variable_memory_list,
                               temp_variable_memory_list,
                               output_name_and_arr_list);
    }

    inline auto make_max_pool_primitive(
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

        auto max_pool_output_md =
          mkldnn::memory::desc({output_dims}, mkldnn::memory::data_type::f32,
                               mkldnn::memory::format::any);

        auto max_pool_desc = mkldnn::pooling_forward::desc(
          mkldnn::prop_kind::forward, mkldnn::pooling_max,
          input_memory.get_primitive_desc().desc(), max_pool_output_md, strides,
          kernel_shape, padding_l, padding_r, mkldnn::padding_kind::zero);
        auto max_pool_pd =
          mkldnn::pooling_forward::primitive_desc(max_pool_desc, engine);

        std::vector<mkldnn::primitive> net;

        auto max_pool_indices_memory =
          mkldnn::memory(max_pool_pd.workspace_primitive_desc());
        temp_variable_memory_list.push_back(max_pool_indices_memory);

        manage_output_memory(required_output_set, output_name, dtype_t::float_,
                             output_dims, input_origin_format,
                             max_pool_pd.dst_primitive_desc(),
                             variable_memory_list, temp_variable_memory_list,
                             output_name_and_arr_list, net, engine,
                             [&input_memory, &max_pool_indices_memory,
                              &max_pool_pd](auto& op_output_memory) {
                                 return mkldnn::pooling_forward(
                                   max_pool_pd, input_memory, op_output_memory,
                                   max_pool_indices_memory);
                             });

        return std::make_tuple(net, variable_memory_list,
                               temp_variable_memory_list,
                               output_name_and_arr_list);
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

        auto const& output_name = node.output(0);

        std::vector<std::pair<
          std::string, std::tuple<mkldnn::memory, mkldnn::memory::format>>>
          variable_memory_list;
        std::vector<mkldnn::memory>
          temp_variable_memory_list; // for temporary memory's life
        std::vector<std::pair<std::string, array>> output_name_and_arr_list;

        std::vector<mkldnn::primitive> net;

        manage_output_memory(
          required_output_set, output_name, dtype_t::float_, input_output_dims,
          input_origin_format, input_memory.get_primitive_desc(),
          variable_memory_list, temp_variable_memory_list,
          output_name_and_arr_list, net, engine,
          [&input_memory](auto& op_output_memory) {
              return mkldnn::reorder(input_memory, op_output_memory);
          });

        return std::make_tuple(net, variable_memory_list,
                               temp_variable_memory_list,
                               output_name_and_arr_list);
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
        auto shape = load_attribute_ints(attribute_table, "shape");

        auto const& input_memory_and_origin_format =
          find_value(variable_memory_table, node.input(0));
        auto const& input_memory = std::get<0>(input_memory_and_origin_format);
        auto input_dims = extract_dims(input_memory);
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

    inline auto make_fc_primitive(
      std::unordered_map<std::string, const mkldnn::memory> const&
        parameter_memory_table,
      std::unordered_map<std::string, std::tuple<const mkldnn::memory,
                                                 mkldnn::memory::format>> const&
        variable_memory_table,
      std::set<std::string> const& required_output_set,
      onnx::NodeProto const& node, mkldnn::engine const& engine) {

        auto attribute_table = instant::make_attribute_table(node);

        auto axis = load_attribute_int(attribute_table, "axis");
        assert(axis == 1);
        auto axis_w = load_attribute_int(attribute_table, "axis_w");
        assert(axis_w == 1);

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
        mkldnn::memory::dims output_dims{input_dims[0], bias_dims[0]};

        auto const& output_name = node.output(0);

        auto fc_input_md =
          mkldnn::memory::desc({input_dims}, mkldnn::memory::data_type::f32,
                               mkldnn::memory::format::any);
        auto fc_weight_md =
          mkldnn::memory::desc({weight_dims}, mkldnn::memory::data_type::f32,
                               mkldnn::memory::format::any);
        auto fc_output_md =
          mkldnn::memory::desc({output_dims}, mkldnn::memory::data_type::f32,
                               mkldnn::memory::format::any);

        mkldnn::inner_product_forward::desc fc_desc(
          mkldnn::prop_kind::forward_inference, fc_input_md, fc_weight_md,
          bias_memory.get_primitive_desc().desc(), fc_output_md);
        auto fc_pd =
          mkldnn::inner_product_forward::primitive_desc(fc_desc, engine);

        std::vector<mkldnn::primitive> net;
        std::vector<mkldnn::memory>
          temp_variable_memory_list; // for temporary memory's life

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

        std::vector<std::pair<
          std::string, std::tuple<mkldnn::memory, mkldnn::memory::format>>>
          variable_memory_list;
        std::vector<std::pair<std::string, array>> output_name_and_arr_list;

        manage_output_memory(
          required_output_set, output_name, dtype_t::float_, output_dims,
          input_origin_format, fc_pd.dst_primitive_desc(), variable_memory_list,
          temp_variable_memory_list, output_name_and_arr_list, net, engine,
          [&fc_pd, &fc_input_memory, &fc_weight_memory,
           &bias_memory](auto& op_output_memory) {
              return mkldnn::inner_product_forward(
                fc_pd, fc_input_memory, fc_weight_memory, bias_memory,
                op_output_memory);
          });

        return std::make_tuple(net, variable_memory_list,
                               temp_variable_memory_list,
                               output_name_and_arr_list);
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

        auto const& output_name = node.output(0);

        std::vector<mkldnn::memory>
          temp_variable_memory_list; // for temporary memory's life

        auto op_desc = mkldnn::softmax_forward::desc(
          mkldnn::prop_kind::forward_inference,
          input_memory.get_primitive_desc().desc(), softmax_axis);
        auto op_pd = mkldnn::softmax_forward::primitive_desc(op_desc, engine);

        std::vector<mkldnn::primitive> net;
        std::vector<std::pair<
          std::string, std::tuple<mkldnn::memory, mkldnn::memory::format>>>
          variable_memory_list;
        std::vector<std::pair<std::string, array>> output_name_and_arr_list;

        manage_output_memory(required_output_set, output_name, dtype_t::float_,
                             input_output_dims, input_origin_format,
                             input_memory.get_primitive_desc(),
                             variable_memory_list, temp_variable_memory_list,
                             output_name_and_arr_list, net, engine,
                             [&input_memory, &op_pd](auto& op_output_memory) {
                                 return mkldnn::softmax_forward(
                                   op_pd, input_memory, op_output_memory);
                             });

        return std::make_tuple(net, variable_memory_list,
                               temp_variable_memory_list,
                               output_name_and_arr_list);
    }

} // namespace instant

#endif // INSTANT_OPERATOR_HPP
