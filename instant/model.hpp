#ifndef INSTANT_MODEL_HPP
#define INSTANT_MODEL_HPP

namespace instant {
    /*
    class model {
    public:
        virtual void run(std::unordered_map<array> const& input_table,
                         std::unordered_map<array> const& output_table) = 0;
    }

    inline auto
    make_model(
      instant::graph const& graph,
      std::unordered_map<std::string, array> const& parameter_table,
      std::vector<std::tuple<std::string, dtype_t, std::vector<int>>> const&
        input_name_dtype_dims_list,
      std::set<std::string> const& required_output_name_set,
      std::function<unique_ptr<model>> const& model_generator) {
        return model_generator.operator()(graph, parameter_table,
                                          input_name_dtype_dims_list,
                                          required_output_name_set);
    }
    */
} // namespace instant

#endif // INSTANT_MODEL_HPP
