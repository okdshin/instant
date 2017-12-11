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
        //std::cout << key << std::endl;
        auto found = m.find(key);
        if(found == m.end()) {
            throw "not found";
        }
        return found->second;
    }

    template <typename Key, typename T>
    inline auto& find_value(std::unordered_map<Key, T>& m, Key const& key) {
        //std::cout << key << std::endl;
        auto found = m.find(key);
        if(found == m.end()) {
            throw "not found";
        }
        return found->second;
    }

    inline auto extract_dims(mkldnn::memory const& m) {
        auto const& d = m.get_primitive_desc().desc().data;
        return std::vector<int>(d.dims, d.dims + d.ndims);
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

        onnx::AttributeProto const& pads_attr =
          find_value(attribute_table, "pads"s);
        assert(pads_attr.ints().size() == 4);
        auto padding_l = mkldnn::memory::dims(pads_attr.ints().begin() + 0,
                                              pads_attr.ints().begin() + 2);
        auto padding_r = mkldnn::memory::dims(pads_attr.ints().begin() + 2,
                                              pads_attr.ints().begin() + 4);

        // Load input and weight
        auto const& input_memory =
          find_value(variable_memory_table, node.input(0));
        auto const& weight_memory =
          find_value(parameter_memory_table, node.input(1));
        auto input_dims = extract_dims(input_memory);
        auto weight_dims = extract_dims(weight_memory);
        auto output_tz = make_conv_output_tz(input_dims, weight_dims, strides,
                                             padding_l, padding_r);
        std::unique_ptr<mkldnn::memory> output_memory_p;
        std::unique_ptr<instant::array> output_arr_p;
        auto const& output_name = node.output(0);
        std::vector<mkldnn::memory>
          temp_variable_memory_list; // for temporary memory's life
        if(output_name_set.find(output_name) != output_name_set.end()) {
            output_arr_p =
              std::make_unique<instant::array>(dtype_t::float_, output_tz);
            output_memory_p =
              std::make_unique<mkldnn::memory>(mkldnn::memory({{{output_tz},
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
            temp_variable_memory_list.push_back(conv_output_memory);
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
            net.push_back(mkldnn::reorder(conv_output_memory, *output_memory_p));
        }

        auto output_name_and_arr =
          std::make_pair(output_name, std::move(*output_arr_p));
        return std::make_tuple(net, conv_output_memory,
                               temp_variable_memory_list,
                               std::vector<decltype(output_name_and_arr)>{
                                 std::move(output_name_and_arr)});
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

    inline auto
    run_model(onnx::GraphProto const& graph,
              std::unordered_map<std::string, const mkldnn::memory> const&
                parameter_memory_table,
              std::unordered_map<std::string, const mkldnn::memory>&
                variable_memory_table,
              std::set<std::string> const& output_name_set,
              instant::context const& context = instant::get_context()) {
        std::unordered_map<std::string, instant::array> output_table;
        std::vector<mkldnn::primitive> net;
        std::vector<mkldnn::memory> temp_variable_memory_list;
        for(auto const& node : graph.node()) {
            if(node.op_type() == "Conv") {
                std::cout << "Conv<\n";
                auto temp_tuple = make_conv_primitive(
                  parameter_memory_table, variable_memory_table,
                  output_name_set, node, context.engine());
                auto const& conv_net = std::get<0>(temp_tuple);
                auto const& output_memory = std::get<1>(temp_tuple);
                auto const& temp_vars = std::get<2>(temp_tuple);
                auto const& output_name_and_arr_list = std::get<3>(temp_tuple);

                net.insert(net.end(), conv_net.begin(), conv_net.end());
                variable_memory_table.insert({node.output(0), output_memory});
                temp_variable_memory_list.insert(
                  temp_variable_memory_list.end(), temp_vars.begin(),
                  temp_vars.end());
                for(auto&& output_name_and_arr : output_name_and_arr_list) {
                    output_table.insert(std::move(output_name_and_arr));
                }
                break; // TODO

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
        std::cout << "net size " << net.size() << std::endl;
        mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();
        return output_table;
    }

} // namespace instant

#endif // INSTANT_MODEL_HPP
