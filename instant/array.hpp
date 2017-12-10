#ifndef INSTANT_ARRAY_HPP
#define INSTANT_ARRAY_HPP

#include <instant/dtype.hpp>
#include <memory>

namespace instant {

    inline std::shared_ptr<void> allocate_data(dtype_t d,
                                                std::vector<int> const& dims) {
        auto total_size =
          std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>());
        if(d == dtype_t::float_) {
            return std::unique_ptr<float[]>(new float[total_size]);
        }
        throw "Not implemented";
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

} // namespace instant

#endif // INSTANT_ARRAY_HPP
