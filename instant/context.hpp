#include <mkldnn.hpp>

namespace instant {
    class context {
    public:
        explicit context(size_t index) : engine_(mkldnn::engine::cpu, index) {}

        auto& engine() const { return engine_; }

    private:
        mkldnn::engine engine_;
    };
} // namespace instant
