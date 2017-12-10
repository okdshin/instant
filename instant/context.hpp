#include <mkldnn.hpp>

namespace instant {
    class context {
    public:
        context() : context(0) {}
        explicit context(size_t index) : engine_(mkldnn::engine::cpu, index) {}

        auto const& engine() const { return engine_; }

    private:
        mkldnn::engine engine_;
    };

    namespace {
        thread_local ::instant::context thread_local_context;
    } // namespace

    inline auto set_context(context const& ctx) { thread_local_context = ctx; }

    inline auto get_context() { return thread_local_context; }

} // namespace instant
