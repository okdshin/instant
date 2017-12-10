#ifndef INSTANT_MODEL_HPP
#define INSTANT_MODEL_HPP

#include <unordered_map>

#include <instant/array.hpp>
#include <instant/context.hpp>
#include <instant/load_onnx.hpp>

#include <mkldnn.hpp>

namespace instant {
    template <typename Key, typename T>
    inline auto const& find_value(std::unordered_map<Key, T> const& m,
                                  Key const& key) {
        std::cout << key << std::endl;
        auto found = m.find(key);
        if(found == m.end()) {
            throw "not found";
        }
        return found->second;
    }

    inline auto make_conv_output_tz(mkldnn::memory::dims const& input_tz,
                                    mkldnn::memory::dims const& weight_tz,
                                    mkldnn::memory::dims const& stride,
                                    mkldnn::memory::dims const& padding_l,
                                    mkldnn::memory::dims const& padding_r) {
        auto calc_length = [](int il, int kl, int pl, int pr, int s) {
            return (il - kl + pl + pr) / s + 1;
        };
        auto batch_size = input_tz[0];
        auto output_channel_num = weight_tz[0];
        auto ih = input_tz[2];
        auto iw = input_tz[3];
        auto kh = weight_tz[2];
        auto kw = weight_tz[3];
        return mkldnn::memory::dims(
          {batch_size, output_channel_num,
           calc_length(ih, kh, padding_l[0], padding_r[0], stride[0]),
           calc_length(iw, kw, padding_l[1], padding_r[1], stride[1])});
    }

    inline auto make_conv_primitive(
      std::unordered_map<std::string, instant::array> const& parameter_table,
      std::unordered_map<std::string, instant::array> const& variable_table,
      onnx::NodeProto const& node, mkldnn::engine const& engine) {
        using namespace std::literals::string_literals;

        auto attribute_table = instant::make_attribute_table(node);

        onnx::AttributeProto const& strides_attr =
          find_value(attribute_table, "strides"s);
        assert(strides_attr.ints().size() == 2);
        auto strides = mkldnn::memory::dims(strides_attr.ints().begin(),
                                            strides_attr.ints().end());

        onnx::AttributeProto const& pads_attr =
          find_value(attribute_table, "pads"s);
        assert(pads_attr.ints().size() == 4);
        auto padding_l = mkldnn::memory::dims(pads_attr.ints().begin() + 0,
                                              pads_attr.ints().begin() + 2);
        auto padding_r = mkldnn::memory::dims(pads_attr.ints().begin() + 2,
                                              pads_attr.ints().begin() + 4);

        auto const& input = find_value(variable_table, node.input(0));
        auto const& weight = find_value(parameter_table, node.input(1));
        mkldnn::memory::dims input_tz(input.dims().begin(), input.dims().end());
        mkldnn::memory::dims weight_tz(weight.dims().begin(),
                                       weight.dims().end());
        auto output_tz = make_conv_output_tz(input_tz, weight_tz, strides,
                                             padding_l, padding_r);
        std::cout << "input_tz ";
        for(auto i : input_tz) {
            std::cout << i << " ";
        }
        std::cout << "\n";
        std::cout << "weight_tz ";
        for(auto i : weight_tz) {
            std::cout << i << " ";
        }
        std::cout << "\n----\n";

        auto user_input_memory =
          mkldnn::memory({{{input_tz},
                           mkldnn::memory::data_type::f32,
                           mkldnn::memory::format::nchw},
                          engine},
                         static_cast<float*>(input.data()));
        auto user_weight_memory =
          mkldnn::memory({{{weight_tz},
                           mkldnn::memory::data_type::f32,
                           mkldnn::memory::format::oihw},
                          engine},
                         static_cast<float*>(weight.data()));

        auto conv_input_md =
          mkldnn::memory::desc({input_tz}, mkldnn::memory::data_type::f32,
                               mkldnn::memory::format::any);
        auto conv_weight_md =
          mkldnn::memory::desc({weight_tz}, mkldnn::memory::data_type::f32,
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
            auto const& bias = find_value(parameter_table, node.input(2));
            mkldnn::memory::dims bias_tz(bias.dims().begin(),
                                         bias.dims().end());
            auto bias_md =
              mkldnn::memory::desc({bias_tz}, mkldnn::memory::data_type::f32,
                                   mkldnn::memory::format::x);
            conv_desc_p = std::make_unique<mkldnn::convolution_forward::desc>(
              mkldnn::prop_kind::forward, mkldnn::algorithm::convolution_direct,
              conv_input_md, conv_weight_md, bias_md, conv_output_md, strides,
              padding_l, padding_r, mkldnn::padding_kind::zero);
        }
        auto conv_pd =
          mkldnn::convolution_forward::primitive_desc(*conv_desc_p, engine);
        std::vector<mkldnn::primitive> net;

        auto conv_input_memory = user_input_memory;
        if(mkldnn::memory::primitive_desc(conv_pd.src_primitive_desc()) !=
           user_input_memory.get_primitive_desc()) {
            conv_input_memory = mkldnn::memory(conv_pd.src_primitive_desc());
            net.push_back(
              mkldnn::reorder(user_input_memory, conv_input_memory));
        }

        auto conv_weight_memory = user_weight_memory;
        if(mkldnn::memory::primitive_desc(conv_pd.weights_primitive_desc()) !=
           user_input_memory.get_primitive_desc()) {
            conv_weight_memory =
              mkldnn::memory(conv_pd.weights_primitive_desc());
            net.push_back(
              mkldnn::reorder(user_weight_memory, conv_input_memory));
        }

        auto conv_output_memory = mkldnn::memory(conv_pd.dst_primitive_desc());

        if(node.input_size() == 2) {
            net.push_back(mkldnn::convolution_forward(
              conv_pd, conv_input_memory, conv_weight_memory,
              conv_output_memory));
        } else {
            auto const& bias = find_value(parameter_table, node.input(2));
            auto conv_bias_memory =
              mkldnn::memory(conv_pd.bias_primitive_desc(), bias.data());
            net.push_back(mkldnn::convolution_forward(
              conv_pd, conv_input_memory, conv_weight_memory, conv_bias_memory,
              conv_output_memory));
        }

        return net;
    }

    inline auto construct_net(
      onnx::GraphProto const& graph,
      std::unordered_map<std::string, instant::array> parameter_table,
      std::unordered_map<std::string, instant::array> variable_table,
      instant::context const& context = instant::get_context()) {
        std::vector<mkldnn::primitive> net;
        for(auto const& node : graph.node()) {
            if(node.op_type() == "Conv") {
                std::cout << "Conv<\n";
                auto conv_net = make_conv_primitive(
                  parameter_table, variable_table, node, context.engine());
                net.insert(net.end(), conv_net.begin(), conv_net.end());

            } else if(node.op_type() == "Gemm") {
                std::cout << "Gemm<\n";
            } else if(node.op_type() == "Relu") {
                std::cout << "Relu<\n";
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
            std::cout << " ";
            for(auto const& i : node.input()) {
                std::cout << i << " ";
            }
            std::cout << "\n-> ";
            for(auto const& o : node.output()) {
                std::cout << o << " ";
            }
            std::cout << "\n";
        }
    }

} // namespace instant

#endif // INSTANT_MODEL_HPP
