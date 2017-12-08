#ifndef INSTANT_ARRAY_HPP
#define INSTANT_ARRAY_HPP

#include <instant/dtype.hpp>
#include <memory>

namespace instant {

    class array {
    public:
        array() = default;
        array(dtype_t d, std::vector<int> const& dims, std::shared_ptr<void> data)
          : dtype_(d), dims_(dims), data_(std::move(data)) {}
        dtype_t dtype() const { return dtype_; }
        auto& dims() const { return dims_; }
        auto* data() const { return data_.get(); }

    private:
        dtype_t dtype_;
        std::vector<int> dims_;
        std::shared_ptr<void> data_;
    };

} // namespace instant

#endif // INSTANT_ARRAY_HPP
