#ifndef INSTANT_GRAPH_HPP
#define INSTANT_GRAPH_HPP

#include <set>
#include <unordered_map>
#include <variant>
#include <vector>

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
        auto input_num() const { return input_name_list_.size(); }
        auto const& input() const { return input_name_list_; }

        auto const& output(int index) const {
            return output_name_list_.at(index);
        }
        auto output_num() const { return output_name_list_.size(); }
        auto const& output() const { return output_name_list_; }

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
        if(a.input_num() != b.input_num()) {
            return a.input_num() < b.input_num();
        }
        for(auto i = 0; i < a.input_num(); ++i) {
            if(a.input(i) != b.input(i)) {
                return a.input(i) < b.input(i);
            }
        }
        if(a.output_num() != b.output_num()) {
            return a.output_num() < b.output_num();
        }
        for(auto i = 0; i < a.output_num(); ++i) {
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
        return mkldnn::memory::dims(
          {batch_size, output_channel_num,
           calc_length(ih, kh, pads[0], pads[2], stride[0]),
           calc_length(iw, kw, pads[1], pads[3], stride[1])});
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

    inline auto extract_needed_nodes_and_input_names(
      std::vector<node> const& node_list,
      std::vector<std::string> required_output_name_list) {
        std::set<node> needed_node_set;
        std::set<std::string> needed_input_name_set;
        while(!required_output_name_list.empty()) {
            std::vector<std::string> next_required_output_name_list;
            for(auto const& required_output_name : required_output_name_list) {
                auto needed_node_iter = std::find(
                  node_list.begin(), node_list.end(), [](auto const& node) {
                      return std::any_of(
                        node.output().begin(), node.output().end(),
                        [&required_output_name](auto const& output_name) {
                            return output_name == required_output_name;
                        });
                  });
                if(needed_node_iter == node_list.end()) {
                    needed_input_name_set.insert(required_output_name);
                } else {
                    needed_node_set.insert(*needed_node_iter);
                    next_required_output_name_list.insert(
                      needed_node_iter->input().begin(),
                      needed_node_iter->input().end());
                }
            }
            required_output_name_list = next_required_output_name_list;
        }
        return std::make_tuple(needed_node_set, needed_input_name_set);
    }

    inline auto make_graph(std::set<node> const& node_set,
                           std::set<std::string> const& input_name_set) {
        auto available_variable_name_set = input_name_set;
        instant::graph graph;
        while(!node_set.empty()) {
            std::set<node> next_node_set;
            auto next_available_variable_name_set = available_variable_name_set;
            std::set<node> current_node_set;
            for(auto const& node : node_set) {
                std::vector<std::string> unavailable_variable_name_list;
                std::set_difference(
                  node.input().begin(), node.input().end(),
                  available_variable_name_set.begin(),
                  available_variable_name_set.end(),
                  std::back_inserter(unavailable_variable_name_list));
                if(unavailable_variable_name_list.empty()) {
                    next_available_variable_name_set.insert(
                      node.output().begin(), node.output().end());
                    current_node_set.insert(node);
                } else {
                    next_node_set.insert(node);
                }
            }
            node_set = next_node_set;
            available_variable_name_set = next_available_variable_name_set;
            graph.push_back(current_node_set);
        }
        return graph;
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

} // namespace instant

#endif // INSTANT_GRAPH_HPP
