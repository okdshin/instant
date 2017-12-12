#ifndef INSTANT_MODEL_HPP
#define INSTANT_MODEL_HPP

#include <functional>
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
      std::unordered_map<std::string, const mkldnn::memory> const&
        variable_memory_table,
      std::set<std::string> const& output_name_set, onnx::NodeProto const& node,
      mkldnn::engine const& engine) {

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
        // assert(pads_attr.ints().size() == 4);
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
            throw std::runtime_error("Not implemented");
        }

        // Load input and weight
        auto const& input_memory =
          find_value(variable_memory_table, node.input(0));
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
        if(output_name_set.find(output_name) != output_name_set.end()) {
            output_arr_p =
              std::make_unique<instant::array>(dtype_t::float_, output_tz);
            output_memory_p = std::make_unique<mkldnn::memory>(
              mkldnn::memory({{{output_tz},
                               mkldnn::memory::data_type::f32,
                               mkldnn::memory::format::nchw},
                              engine},
                             output_arr_p->data()));
            temp_variable_memory_list.push_back(*output_memory_p);
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
              mkldnn::prop_kind::forward, mkldnn::algorithm::convolution_direct,
              conv_input_md, conv_weight_md, conv_output_md, strides, padding_l,
              padding_r, mkldnn::padding_kind::zero);
        } else {
            auto const& bias_memory =
              find_value(parameter_memory_table, node.input(2));
            conv_desc_p = std::make_unique<mkldnn::convolution_forward::desc>(
              mkldnn::prop_kind::forward, mkldnn::algorithm::convolution_direct,
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
            // temp_variable_memory_list.push_back(conv_output_memory); //
            // useless?
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

        auto output_name_and_mem =
          std::make_pair(output_name, std::move(conv_output_memory));
        auto output_name_and_arr =
          std::make_pair(output_name, std::move(*output_arr_p));
        return std::make_tuple(net,
                               std::vector<decltype(output_name_and_mem)>{
                                 std::move(output_name_and_mem)},
                               temp_variable_memory_list,
                               std::vector<decltype(output_name_and_arr)>{
                                 std::move(output_name_and_arr)});
    }

    inline auto make_relu_primitive(
      std::unordered_map<std::string, const mkldnn::memory> const&
        parameter_memory_table,
      std::unordered_map<std::string, const mkldnn::memory> const&
        variable_memory_table,
      std::set<std::string> const& output_name_set, onnx::NodeProto const& node,
      mkldnn::engine const& engine) {
        auto negative_slope = 0.; // 1.0;
        auto const& input_memory =
          find_value(variable_memory_table, node.input(0));
        auto input_output_dims = extract_dims(input_memory);

        std::unique_ptr<mkldnn::memory> output_memory_p;
        std::unique_ptr<instant::array> output_arr_p;
        auto const& output_name = node.output(0);
        std::vector<mkldnn::memory>
          temp_variable_memory_list; // for temporary memory's life
        if(output_name_set.find(output_name) != output_name_set.end()) {
            output_arr_p = std::make_unique<instant::array>(dtype_t::float_,
                                                            input_output_dims);
            output_memory_p = std::make_unique<mkldnn::memory>(mkldnn::memory(
              input_memory.get_primitive_desc(), output_arr_p->data()));
            temp_variable_memory_list.push_back(*output_memory_p);
        }

        auto relu_desc = mkldnn::eltwise_forward::desc(
          mkldnn::prop_kind::forward, mkldnn::algorithm::eltwise_relu,
          input_memory.get_primitive_desc().desc(), negative_slope);
        auto relu_pd =
          mkldnn::eltwise_forward::primitive_desc(relu_desc, engine);

        std::vector<mkldnn::primitive> net;

        auto relu_output_memory =
          output_memory_p ? *output_memory_p
                          : mkldnn::memory(relu_pd.dst_primitive_desc());
        if(output_memory_p &&
           mkldnn::memory::primitive_desc(relu_pd.dst_primitive_desc()) !=
             output_memory_p->get_primitive_desc()) {
            relu_output_memory = mkldnn::memory(relu_pd.dst_primitive_desc());
            // temp_variable_memory_list.push_back(relu_output_memory); //
            // useless?
        }

        net.push_back(
          mkldnn::eltwise_forward(relu_pd, input_memory, relu_output_memory));

        if(output_memory_p && relu_output_memory != *output_memory_p) {
            std::cout << "reorder" << std::endl;
            net.push_back(
              mkldnn::reorder(relu_output_memory, *output_memory_p));
        }

        auto output_name_and_mem =
          std::make_pair(output_name, std::move(relu_output_memory));
        auto output_name_and_arr =
          std::make_pair(output_name, std::move(*output_arr_p));
        return std::make_tuple(net,
                               std::vector<decltype(output_name_and_mem)>{
                                 std::move(output_name_and_mem)},
                               temp_variable_memory_list,
                               std::vector<decltype(output_name_and_arr)>{
                                 std::move(output_name_and_arr)});
    }

    /*
    inline auto make_max_pool_primitive(
      std::unordered_map<std::string, const mkldnn::memory> const&
        parameter_memory_table,
      std::unordered_map<std::string, const mkldnn::memory> const&
        variable_memory_table,
      std::set<std::string> const& output_name_set, onnx::NodeProto const& node,
      mkldnn::engine const& engine) {
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
        auto kernel_shape =
    mkldnn::memory::dims(kernel_shape_attr.ints().begin(),
                                            kernel_shape_attr.ints().end());

        onnx::AttributeProto const& pads_attr =
          find_value(attribute_table, "pads"s);
        // Load input and weight
        auto const& input_memory =
          find_value(variable_memory_table, node.input(0));
        auto input_dims = extract_dims(input_memory);
        auto output_channel_num = input_dims[1];
        auto output_tz = make_conv_output_tz(input_dims, output_channel_num,
    kernel_shape, strides, padding_l, padding_r);
        std::unique_ptr<mkldnn::memory> output_memory_p;
        std::unique_ptr<instant::array> output_arr_p;
        auto const& output_name = node.output(0);
        std::vector<mkldnn::memory>
          temp_variable_memory_list; // for temporary memory's life
        if(output_name_set.find(output_name) != output_name_set.end()) {
            output_arr_p = std::make_unique<instant::array>(dtype_t::float_,
                                                            input_output_dims);
            output_memory_p = std::make_unique<mkldnn::memory>(mkldnn::memory(
              input_memory.get_primitive_desc(), output_arr_p->data()));
            temp_variable_memory_list.push_back(*output_memory_p);
        }

        auto pool_input_memory = input_memory;
        if(mkldnn::memory::primitive_desc(conv_pd.src_primitive_desc()) !=
           input_memory.get_primitive_desc()) {
            conv_input_memory = mkldnn::memory(conv_pd.src_primitive_desc());
            temp_variable_memory_list.push_back(conv_input_memory);
            net.push_back(mkldnn::reorder(input_memory, conv_input_memory));
        }

    }
    */

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
            } else {
                // TODO
                // throw "Not implemented";
            }
        }
        return memory_table;
    }

    inline auto make_variable_memory_table(
      std::vector<std::tuple<std::string, instant::array,
                             mkldnn::memory::format>>& input_list,
      mkldnn::engine const& engine) {
        std::unordered_map<std::string, const mkldnn::memory> memory_table;
        for(auto& input : input_list) {
            auto& arr = std::get<1>(input);
            auto format = std::get<2>(input);
            mkldnn::memory::dims tz(arr.dims().begin(), arr.dims().end());
            auto mem = mkldnn::memory(
              {{{tz}, mkldnn::memory::data_type::f32, format}, engine},
              arr.data());
            auto const& name = std::get<0>(input);
            memory_table.insert({name, mem});
        }
        return memory_table;
    }

    using primitive_factory = std::function<std::tuple<
      std::vector<mkldnn::primitive>,
      std::vector<std::pair<std::string, mkldnn::memory>>,
      std::vector<mkldnn::memory>, std::vector<std::pair<std::string, array>>>(
      std::unordered_map<std::string, const mkldnn::memory> const&,
      std::unordered_map<std::string, const mkldnn::memory> const&,
      std::set<std::string> const& output_name_set, onnx::NodeProto const&,
      mkldnn::engine const&)>;

    inline auto make_default_primitive_factory_table() {
        std::unordered_map<std::string, primitive_factory>
          primitive_factory_table;
        primitive_factory_table.insert({"Conv", make_conv_primitive});
        // TODO
        // primitive_factory_table.insert({"Relu", make_relu_primitive});
        // primitive_factory_table.insert({"MaxPool", make_max_pool_primitive});
        // primitive_factory_table.insert({"FC", make_max_fc_primitive});
        // primitive_factory_table.insert({"Reshape", make_reshape_primitive});
        // primitive_factory_table.insert({"Dropout", make_dropout_primitive});
        // primitive_factory_table.insert({"Softmax", make_softmax_primitive});
        return primitive_factory_table;
    }

    inline auto
    run_model(onnx::GraphProto const& graph,
              std::unordered_map<std::string, const mkldnn::memory> const&
                parameter_memory_table,
              std::unordered_map<std::string, const mkldnn::memory>&
                variable_memory_table,
              std::set<std::string> const& output_name_set,
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
                    output_name_set, node, context.engine());
                auto& net = std::get<0>(temp_tuple);
                auto& output_name_and_memory_list = std::get<1>(temp_tuple);
                auto& temp_vars = std::get<2>(temp_tuple);
                auto& output_name_and_arr_list = std::get<3>(temp_tuple);

                nets.insert(nets.end(), net.begin(), net.end());
                for(auto&& output_name_and_memory :
                    output_name_and_memory_list) {
                    variable_memory_table.insert(
                      std::move(output_name_and_memory));
                }
                temp_variable_memory_list.insert(
                  temp_variable_memory_list.end(), temp_vars.begin(),
                  temp_vars.end());
                for(auto&& output_name_and_arr : output_name_and_arr_list) {
                    output_table.insert(std::move(output_name_and_arr));
                }
            } catch(std::exception const& e) {
                std::cout << "Error: " << e.what() << std::endl;
            }
        }
        std::cout << "float size " << sizeof(float) << std::endl;
        std::cout << "net size " << nets.size() << std::endl;
        mkldnn::stream(mkldnn::stream::kind::eager).submit(nets).wait();
        return output_table;
        /*
if(node.op_type() == "Conv") {
    std::cout << "Conv<\n";
    try {
        auto temp_tuple = make_conv_primitive(
          parameter_memory_table, variable_memory_table,
          output_name_set, node, context.engine());
    } catch(...) {}
    // break; // TODO

} else if(node.op_type() == "Gemm") {
    std::cout << "Gemm<\n";
} else if(node.op_type() == "Relu") {
    std::cout << "Relu<\n";
    auto temp_tuple = make_relu_primitive(
      parameter_memory_table, variable_memory_table,
      output_name_set, node, context.engine());
    );
} else if(node.op_type() == "Softmax") {
    std::cout << "Softmax<\n";
} else if(node.op_type() == "MaxPool") {
    std::cout << "MaxPool<\n";
} else if(node.op_type() == "Reshape") {
    std::cout << "Reshape<\n";
} else if(node.op_type() == "LRN") {
    std::cout << "LRN<\n";
} else {
    std::cout << node.op_type() << "!!!!!" << std::endl;
    // throw "not implemented"; // TODO
}
*/
        /*
        std::cout << " ";
        for(auto const& i : node.input()) {
            std::cout << i << " ";
        }
        std::cout << "\n-> ";
        for(auto const& o : node.output()) {
            std::cout << o << " ";
        }
        std::cout << "\n";
        */
    }

} // namespace instant
#endif // INSTANT_MODEL_HPP
