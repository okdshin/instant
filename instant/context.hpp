#ifndef INSTANT_CONTEXT_HPP
#define INSTANT_CONTEXT_HPP

#include <mkldnn.hpp>

namespace instant {
    class context {
    public:
        context() : context(0) {}
        explicit context(int cpu_id)
          : cpu_id_(cpu_id), engine_(mkldnn::engine::cpu, cpu_id) {}

        auto cpu_id() const { return cpu_id_; }
        auto const& engine() const { return engine_; }

    private:
        int cpu_id_;
        mkldnn::engine engine_;
    };

    namespace {
        thread_local ::instant::context thread_local_context;
    } // namespace

    inline auto set_context(context const& ctx) { thread_local_context = ctx; }

    inline auto get_context() { return thread_local_context; }

    inline auto get_available_cpu_count() {
        return mkldnn::engine::get_count(mkldnn::engine::cpu);
    }

    class scoped_context {
    public:
        scoped_context(int cpu_id) : prev_context_(::instant::get_context()) {
            if(get_available_cpu_count() <= cpu_id) {
                throw std::runtime_error("Invalid cpu id: " +
                                         std::to_string(cpu_id));
            }
            ::instant::set_context(context(cpu_id));
        }
        ~scoped_context() { ::instant::set_context(prev_context_); }

    private:
        context prev_context_;
    };

} // namespace instant

#endif // INSTANT_CONTEXT_HPP
