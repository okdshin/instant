#ifndef INSTANT_ENGINE_HPP
#define INSTANT_ENGINE_HPP

namespace instant {
    mkldnn::engine const& thread_local_default_engine() {
        thread_local mkldnn::engine default_engine;
        return default_engine;
    }

} // namespace instant


#endif // INSTANT_ENGINE_HPP
