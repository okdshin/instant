#ifndef INSTANT_GRAPH_HPP
#define INSTANT_GRAPH_HPP

#include <functional>
#include <set>
#include <unordered_map>
#include <variant>
#include <vector>

#include <instant/array.hpp>
#include <instant/op_type.hpp>
#include <instant/utility.hpp>

namespace instant {

    using attribute =
      std::variant<int, float, std::vector<int>, std::vector<float>>;

    class node {
    public:
        node(op_type_t op_type, std::vector<std::string> const& input_name_list,
             std::vector<std::string> const& output_name_list,
             std::unordered_map<std::string, attribute> const& attribute_table)
          : op_type_(op_type), input_name_list_(input_name_list),
            output_name_list_(output_name_list),
            attribute_table_(attribute_table) {}

        auto op_type() const { return op_type_; }

        auto const& input(int index) const {
            return input_name_list_.at(index);
        }
        auto const& input() const { return input_name_list_; }
        auto& input() { return input_name_list_; }

        auto const& output(int index) const {
            return output_name_list_.at(index);
        }
        auto const& output() const { return output_name_list_; }
        auto& output() { return output_name_list_; }

        template <typename AttributeType>
        auto const& attribute(std::string const& attr_name) const {
            return std::get<AttributeType>(
              find_value(attribute_table_, attr_name));
        }

    private:
        op_type_t op_type_;
        std::vector<std::string> input_name_list_;
        std::vector<std::string> output_name_list_;
        std::unordered_map<std::string, instant::attribute> attribute_table_;
    };

    inline auto operator<(node const& a, node const& b) {
        if(a.input().size() != b.input().size()) {
            return a.input().size() < b.input().size();
        }
        for(auto i = 0; i < a.input().size(); ++i) {
            if(a.input(i) != b.input(i)) {
                return a.input(i) < b.input(i);
            }
        }
        if(a.output().size() != b.output().size()) {
            return a.output().size() < b.output().size();
        }
        for(auto i = 0; i < a.output().size(); ++i) {
            if(a.output(i) != b.output(i)) {
                return a.output(i) < b.output(i);
            }
        }
        throw std::runtime_error("Do not come here");
    }

    inline auto const& attribute_int(node const& n,
                                     std::string const& attr_name) {
        return n.attribute<int>(attr_name);
    }
    inline auto const& attribute_float(node const& n,
                                       std::string const& attr_name) {
        return n.attribute<float>(attr_name);
    }
    inline auto const& attribute_ints(node const& n,
                                      std::string const& attr_name) {
        return n.attribute<std::vector<int>>(attr_name);
    }
    inline auto const& attribute_floats(node const& n,
                                        std::string const& attr_name) {
        return n.attribute<std::vector<float>>(attr_name);
    }
    inline auto attributes_for_2d_data_processing(node const& n) {
        // Workaround for onnx-chainer. see
        // https://github.com/chainer/onnx-chainer/pull/11
        auto pads = attribute_ints(n, "pads");
        if(pads.size() == 2) {
            pads = std::vector{pads[0], pads[1], pads[0], pads[1]};
        }
        return std::make_tuple(attribute_ints(n, "strides"),
                               attribute_ints(n, "kernel_shape"), pads);
    }

    using graph = std::vector<std::set<node>>;

    inline auto calc_2d_output_dims(std::vector<int> const& input_dims,
                                    int output_channel_num,
                                    std::vector<int> const& kernel_shape,
                                    std::vector<int> const& strides,
                                    std::vector<int> const& pads) {
        if(pads.size() != 4) {
            throw std::runtime_error("pads size is invalid (expected 4 but " +
                                     std::to_string(pads.size()) + ")");
        }
        auto calc_length = [](int il, int kl, int p_begin, int p_end, int s) {
            return (il - kl + p_begin + p_end) / s + 1;
        };
        auto batch_size = input_dims[0];
        auto ih = input_dims[2];
        auto iw = input_dims[3];
        auto kh = kernel_shape[0];
        auto kw = kernel_shape[1];
        return std::vector<int>(
          {batch_size, output_channel_num,
           calc_length(ih, kh, pads[0], pads[2], strides[0]),
           calc_length(iw, kw, pads[1], pads[3], strides[1])});
    }

    inline auto
    calc_2d_output_dims(instant::node const& node, int output_channel_num,
                        std::unordered_map<std::string, std::vector<int>> const&
                          variable_dims_table) {
        return calc_2d_output_dims(
          find_value(variable_dims_table, node.input(0)), output_channel_num,
          attribute_ints(node, "kernel_shape"), attribute_ints(node, "strides"),
          attribute_ints(node, "pads"));
    }

    inline auto
    get_batch_size_from_variable_dims(std::vector<int> const& variable_dims) {
        return variable_dims.at(0); // n of nchw
    }

    inline auto
    get_channel_num_from_variable_dims(std::vector<int> const& variable_dims) {
        return variable_dims.at(1); // c of nchw
    }

    inline auto get_output_channel_num_from_parameter_dims(
      std::vector<int> const& parameter_dims) {
        return parameter_dims.at(0); // o of oihw
    }

    inline auto
    extract_needed_node_list(std::vector<node> const& node_list,
                             std::set<std::string> required_output_name_set) {
        std::vector<node> needed_node_list;
        while(!required_output_name_set.empty()) {
            std::set<std::string> next_required_output_name_set;
            for(auto const& required_output_name : required_output_name_set) {
                // Search node that issues required output
                auto needed_node_iter = std::find_if(
                  node_list.begin(), node_list.end(),
                  [&required_output_name](auto const& node) {
                      return std::any_of(
                        node.output().begin(), node.output().end(),
                        [&required_output_name](auto const& output_name) {
                            return output_name == required_output_name;
                        });
                  });
                if(needed_node_iter != node_list.end()) {
                    needed_node_list.push_back(*needed_node_iter);
                    next_required_output_name_set.insert(
                      needed_node_iter->input().begin(),
                      needed_node_iter->input().end());
                }
            }
            required_output_name_set = next_required_output_name_set;
        }
        return needed_node_list;
    }

    inline auto make_graph(std::vector<node> node_list,
                           std::set<std::string> const& given_input_name_set,
                           std::set<std::string> const& parameter_name_set) {
        std::set<node> node_set(node_list.begin(), node_list.end());
        auto available_value_name_set = given_input_name_set;
        available_value_name_set.insert(parameter_name_set.begin(),
                                        parameter_name_set.end());
        instant::graph graph;
        while(!node_set.empty()) {
            std::set<node> next_node_set;
            auto next_available_value_name_set = available_value_name_set;
            std::set<node> current_node_set;
            for(auto const& node : node_set) {
                std::vector<std::string> unavailable_value_name_list;
                std::set<std::string> input_name_set(node.input().begin(),
                                                     node.input().end());
                std::set_difference(
                  input_name_set.begin(), input_name_set.end(),
                  available_value_name_set.begin(),
                  available_value_name_set.end(),
                  std::back_inserter(unavailable_value_name_list));
                if(unavailable_value_name_list.empty()) {
                    next_available_value_name_set.insert(node.output().begin(),
                                                         node.output().end());
                    current_node_set.insert(node);
                } else {
                    next_node_set.insert(node);
                }
            }
            node_set = next_node_set;
            available_value_name_set = next_available_value_name_set;
            graph.push_back(current_node_set);
        }
        return graph;
    }

    inline auto
    extract_needed_input_name_set(std::vector<node> const& node_list,
                                  std::set<std::string> parameter_name_set) {
        std::set<std::string> input_name_set;
        for(auto const& node : node_list) {
            input_name_set.insert(node.input().begin(), node.input().end());
            parameter_name_set.insert(node.output().begin(),
                                      node.output().end());
        }
        std::set<std::string> needed_input_name_set;
        std::set_difference(
          input_name_set.begin(), input_name_set.end(),
          parameter_name_set.begin(), parameter_name_set.end(),
          std::inserter(needed_input_name_set, needed_input_name_set.end()));
        return needed_input_name_set;
    }

    inline auto extract_needed_parameter_name_set(
      std::vector<node> const& node_list,
      std::set<std::string> given_input_name_set) {
        std::set<std::string> input_name_set;
        for(auto const& node : node_list) {
            input_name_set.insert(node.input().begin(), node.input().end());
            given_input_name_set.insert(node.output().begin(),
                                        node.output().end());
        }
        std::set<std::string> needed_parameter_name_set;
        std::set_difference(input_name_set.begin(), input_name_set.end(),
                            given_input_name_set.begin(),
                            given_input_name_set.end(),
                            std::inserter(needed_parameter_name_set,
                                          needed_parameter_name_set.end()));
        return needed_parameter_name_set;
    }

    inline auto make_variable_dims_table(
      instant::graph const& graph,
      std::unordered_map<std::string, array> const& parameter_table,
      std::unordered_map<std::string, std::vector<int>> input_dims_table) {
        auto variable_dims_table = input_dims_table;
        for(auto const& node_set : graph) {
            for(auto const& node : node_set) {
                if(node.op_type() == op_type_t::conv) {
                    auto weight_name = node.input(1);
                    auto output_channel_num =
                      get_output_channel_num_from_parameter_dims(
                        find_value(parameter_table, weight_name).dims());
                    auto output_dims = calc_2d_output_dims(
                      node, output_channel_num, variable_dims_table);
                    variable_dims_table.insert({node.output(0), output_dims});
                } else if(node.op_type() == op_type_t::max_pool) {
                    auto input_name = node.input(0);
                    auto output_channel_num =
                      get_channel_num_from_variable_dims(
                        find_value(variable_dims_table, input_name));
                    auto output_dims = calc_2d_output_dims(
                      node, output_channel_num, variable_dims_table);
                    variable_dims_table.insert({node.output(0), output_dims});
                } else if(node.op_type() == op_type_t::fc) {
                    auto input_name = node.input(0);
                    auto batch_size = get_batch_size_from_variable_dims(
                      find_value(variable_dims_table, input_name));
                    auto bias_size = get_output_channel_num_from_parameter_dims(
                      find_value(parameter_table, node.input(2)).dims());
                    std::vector output_dims{batch_size, bias_size};
                    variable_dims_table.insert({node.output(0), output_dims});
                } else if(node.op_type() == op_type_t::reshape) {
                    auto output_dims = attribute_ints(node, "shape");
                    variable_dims_table.insert({node.output(0), output_dims});
                } else {
                    auto input_name = node.input(0);
                    auto output_dims =
                      find_value(variable_dims_table, input_name);
                    variable_dims_table.insert({node.output(0), output_dims});
                }
            }
        }
        return variable_dims_table;
    }

    inline auto
    extract_node_ref_that_has_specific_output(std::vector<node>& node_list,
                                              std::string const& output_name) {
        for(auto& node : node_list) {
            auto node_iter = std::find(node.output().begin(),
                                       node.output().end(), output_name);
            if(node_iter != node.output().end()) {
                return std::ref(node);
            }
        }
        throw std::runtime_error("No node has output named: " + output_name);
    }

    inline auto extract_node_ref_list_that_has_specific_input(
      std::vector<node>& node_list, std::string const& input_name) {
        std::vector<std::reference_wrapper<node>> node_ref_list;
        for(auto& node : node_list) {
            auto input_iter =
              std::find(node.input().begin(), node.input().end(), input_name);
            if(input_iter != node.input().end()) {
                node_ref_list.push_back(std::ref(node));
            }
        }
        return node_ref_list;
    }

    template <typename Visitor>
    auto reconstruct_node_list(std::vector<node>& node_list, Visitor visitor) {
        auto node_list_for_loop = node_list;
        for(auto& node : node_list_for_loop) {
            visitor(node_list, node);
        }
    }

    inline auto trim_node(std::vector<node>& node_list,
                          instant::node const& node) {
        auto next_node_ref_list = extract_node_ref_list_that_has_specific_input(
          node_list, node.output(0));
        for(instant::node& next_node : next_node_ref_list) {
            auto input_name_iter =
              std::find(next_node.input().begin(), next_node.input().end(),
                        node.output(0));
            assert(input_name_iter != next_node.input().end());
            *input_name_iter = node.input(0);
        }
    }

    inline auto trim_dropout(std::vector<node>& node_list,
                             instant::node const& node) {
        if(node.op_type() == op_type_t::dropout) {
            trim_node(node_list, node);
        }
    }

    inline auto trim_reshape(std::vector<node>& node_list,
                             instant::node const& node) {
        if(node.op_type() == op_type_t::reshape) {
            instant::node& prev_node =
              extract_node_ref_that_has_specific_output(node_list,
                                                        node.input(0));
            std::cout << op_type_to_string(prev_node.op_type()) << std::endl;
            //TODO Relu
            if(prev_node.op_type() == op_type_t::conv ||
               prev_node.op_type() == op_type_t::max_pool) {
                auto next_node_ref_set =
                  extract_node_ref_list_that_has_specific_input(node_list,
                                                                node.output(0));
                if(std::all_of(next_node_ref_set.begin(),
                               next_node_ref_set.end(),
                               [](instant::node const& node) {
                                   return node.op_type() == op_type_t::fc;
                               })) { // TODO check shape
                    trim_node(node_list, node);
                }
            }
        }
    }

} // namespace instant

#endif // INSTANT_GRAPH_HPP
