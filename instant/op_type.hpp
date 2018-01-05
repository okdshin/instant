#ifndef INSTANT_OP_TYPE_HPP
#define INSTANT_OP_TYPE_HPP

namespace instant {

    enum class op_type_t {
        conv,
        max_pool,
        fc,
        reshape,
        batch_normalization,
        relu,
        dropout,
        softmax
    };

    inline auto op_type_to_string(op_type_t op_type) {
        if(op_type == op_type_t::conv) {
            return "Conv";
        }
        if(op_type == op_type_t::max_pool) {
            return "MaxPool";
        }
        if(op_type == op_type_t::fc) {
            return "FC";
        }
        if(op_type == op_type_t::reshape) {
            return "Reshape";
        }
        if(op_type == op_type_t::batch_normalization) {
            return "BatchNormalization";
        }
        if(op_type == op_type_t::relu) {
            return "Relu";
        }
        if(op_type == op_type_t::dropout) {
            return "Dropout";
        }
        if(op_type == op_type_t::softmax) {
            return "Softmax";
        }
        throw std::runtime_error("Not come here");
    }

    inline auto string_to_op_type(std::string const& op_str) {
        if(op_str == "Conv") {
            return op_type_t::conv;
        }
        if(op_str == "MaxPool") {
            return op_type_t::max_pool;
        }
        if(op_str == "FC") {
            return op_type_t::fc;
        }
        if(op_str == "Reshape") {
            return op_type_t::reshape;
        }
        if(op_str == "BatchNormalization") {
            return op_type_t::batch_normalization;
        }
        if(op_str == "Relu") {
            return op_type_t::relu;
        }
        if(op_str == "Dropout") {
            return op_type_t::dropout;
        }
        if(op_str == "Softmax") {
            return op_type_t::softmax;
        }
        throw std::runtime_error("Not implemented: "+op_str);
    }

} // namespace instant

#endif // INSTANT_OP_TYPE_HPP
