#ifndef INSTANT_MKLDNN_UTILITY_HPP
#define INSTANT_MKLDNN_UTILITY_HPP

#include <vector>

#include <mkldnn.hpp>

#include <instant/array.hpp>

namespace instant::mkldnn_backend {

    inline auto extract_dims(mkldnn::memory const& m) {
        auto const& d = m.get_primitive_desc().desc().data;
        return std::vector<int>(d.dims, d.dims + d.ndims);
    }

    inline auto dtype_to_mkldnn_memory_data_type(dtype_t dtype) {
        if(dtype == dtype_t::float_) {
            return mkldnn::memory::data_type::f32;
        }
        throw std::runtime_error("Not implemented: " + dtype_to_string(dtype));
    }

    inline auto array_to_memory(array const& arr, mkldnn::memory::format format,
                                mkldnn::engine const& engine) {
        return mkldnn::memory({{{arr.dims()},
                                dtype_to_mkldnn_memory_data_type(arr.dtype()),
                                format},
                               engine},
                              const_cast<void*>(arr.data()));
    }

} // namespace instant::mkldnn_backend

#endif // INSTANT_MKLDNN_UTILITY_HPP
