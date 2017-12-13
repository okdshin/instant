#ifndef INSTANT_ARRAY_HPP
#define INSTANT_ARRAY_HPP
#include <algorithm>
#include <memory>
#include <numeric>
#include <vector>

#include <instant/dtype.hpp>

namespace instant {

    inline auto calc_total_size(std::vector<int> const& dims) {
        return std::accumulate(dims.begin(), dims.end(), 1,
                               std::multiplies<>());
    }

    inline std::shared_ptr<void> allocate_data(dtype_t d,
                                               std::vector<int> const& dims) {
        auto total_size = calc_total_size(dims);
        if(d == dtype_t::float_) {
            return std::unique_ptr<float[]>(new float[total_size]);
        }
        throw std::runtime_error("Not implemented dtype: " +
                                 std::to_string(static_cast<int>(d)));
    }

    class array {
    public:
        array() = default;
        array(dtype_t d, std::vector<int> const& dims,
              std::shared_ptr<void> data)
          : dtype_(d), dims_(dims), data_(std::move(data)) {}
        array(dtype_t d, std::vector<int> const& dims)
          : array(d, dims, allocate_data(d, dims)) {}

        dtype_t dtype() const { return dtype_; }
        auto const& dims() const { return dims_; }
        auto* data() { return data_.get(); }
        auto const* data() const { return data_.get(); }

    private:
        dtype_t dtype_;
        std::vector<int> dims_;
        std::shared_ptr<void> data_;
    };

    inline auto total_size(array const& a) { return calc_total_size(a.dims()); }

    inline float const* fbegin(array const& a) {
        if(a.dtype() != dtype_t::float_) {
            throw std::runtime_error(
              "fbegin is called but dtype is not float");
        }
        return static_cast<float const*>(a.data());
    }
    inline float const* fend(array const& a) {
        if(a.dtype() != dtype_t::float_) {
            throw std::runtime_error(
              "fend is called but dtype is not float");
        }
        return fbegin(a) + total_size(a);
    }

    inline float* fbegin(array& a) {
        return const_cast<float*>(fbegin(const_cast<array const&>(a)));
    }
    inline float* fend(array& a) {
        return const_cast<float*>(fend(const_cast<array const&>(a)));
    }

    inline float const& fat(array const& a, std::size_t i) {
        if(a.dtype() != dtype_t::float_) {
            throw std::runtime_error(
              "fat is called but dtype is not float");
        }
        return *(static_cast<float const*>(a.data())+i);
    }
    inline float& fat(array& a, std::size_t i) {
        return const_cast<float&>(fat(const_cast<array const&>(a), i));
    }

    template <typename T>
    inline auto uniforms(dtype_t d, std::vector<int> const& dims, T val) {
        auto arr = array(d, dims);
        if(d == dtype_t::float_) {
            std::fill_n(static_cast<float*>(arr.data()), calc_total_size(dims),
                        static_cast<float>(val));
            return arr;
        }
        throw "Not implemented";
    }

    inline auto zeros(dtype_t d, std::vector<int> const& dims) {
        return uniforms(d, dims, 0.);
    }

} // namespace instant

#endif // INSTANT_ARRAY_HPP
