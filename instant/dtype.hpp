#ifndef INSTANT_DTYPE_HPP
#define INSTANT_DTYPE_HPP
#include <cassert>
#include <climits>
#include <cstdint>
#include <onnx.pb.h>
namespace instant {

    //check byte is 8bit
    static_assert(CHAR_BIT == 8, "");

    // check float is 32bit
    static_assert(std::numeric_limits<float>::is_iec559, "");

    enum class dtype_t {
        undefined = onnx::TensorProto_DataType_UNDEFINED,
        float_ = onnx::TensorProto_DataType_FLOAT,
        uint8 = onnx::TensorProto_DataType_UINT8,
        int8 = onnx::TensorProto_DataType_INT8,
        uint16 = onnx::TensorProto_DataType_UINT16,
        int16 = onnx::TensorProto_DataType_INT16,
        int32 = onnx::TensorProto_DataType_INT32,
        int64 = onnx::TensorProto_DataType_INT64,
        string_ = onnx::TensorProto_DataType_STRING,
        bool_ = onnx::TensorProto_DataType_BOOL

        // Advanced
        /*
        float16;
        double_;
        uint32;
        uint64;
        complex64;
        complex128;
        */
    };

    inline auto tensor_proto_data_type_to_dtype_t(onnx::TensorProto_DataType d) {
        return static_cast<instant::dtype_t>(d);
    }

    inline auto dtype_t_to_tensor_proto_data_type(dtype_t d) {
        return static_cast<int>(d);
    }

    template<dtype_t> constexpr int size_in_bytes = 0;

    template<> constexpr int size_in_bytes<dtype_t::float_> = 4;
    template<> constexpr int size_in_bytes<dtype_t::uint8> = 1;
    template<> constexpr int size_in_bytes<dtype_t::int8> = 1;
    template<> constexpr int size_in_bytes<dtype_t::uint16> = 2;
    template<> constexpr int size_in_bytes<dtype_t::int16> = 2;
    template<> constexpr int size_in_bytes<dtype_t::int32> = 4;
    template<> constexpr int size_in_bytes<dtype_t::int64> = 8;
    template<> constexpr int size_in_bytes<dtype_t::string_> = 1; // TODO check size
    template<> constexpr int size_in_bytes<dtype_t::bool_> = 1;

    template<dtype_t> struct dtype_t_to_type {};

    template<> struct dtype_t_to_type<dtype_t::float_> { using type = float; };
    template<> struct dtype_t_to_type<dtype_t::uint8> { using type = std::uint8_t; };
    template<> struct dtype_t_to_type<dtype_t::int8> { using type = std::int8_t; };
    template<> struct dtype_t_to_type<dtype_t::uint16> { using type = std::uint16_t; };
    template<> struct dtype_t_to_type<dtype_t::int16> { using type = std::int16_t; };
    template<> struct dtype_t_to_type<dtype_t::int32> { using type = std::int32_t; };
    template<> struct dtype_t_to_type<dtype_t::int64> { using type = std::int64_t; };
    template<> struct dtype_t_to_type<dtype_t::bool_> { using type = bool; };

    template<dtype_t d>
    using dtype_t_to_type_t = typename dtype_t_to_type<d>::type;


} // namespace instant

#endif //INSTANT_DTYPE_HPP
