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

        auto input(int index) const { return input_name_list_.at(index); }
        auto input_num() const { return input_name_list_.size(); }
        auto output(int index) const { return output_name_list_.at(index); }
        auto output_num() const { return output_name_list_.size(); }

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

} // namespace instant

#endif // INSTANT_GRAPH_HPP
