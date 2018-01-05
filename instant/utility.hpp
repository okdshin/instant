#ifndef INSTANT_UTILITY_HPP
#define INSTANT_UTILITY_HPP

#include <exception>
#include <string>
#include <unordered_map>

namespace instant {

    template <typename T>
    auto const& find_value(std::unordered_map<std::string, T> const& m,
                           std::string const& key) {
        // std::cout << key << std::endl;
        auto found = m.find(key);
        if(found == m.end()) {
            throw std::runtime_error("not found: " + key);
        }
        return found->second;
    }

    template <typename T>
    auto& find_value(std::unordered_map<std::string, T>& m,
                     std::string const& key) {
        // std::cout << key << std::endl;
        auto found = m.find(key);
        if(found == m.end()) {
            throw std::runtime_error("not found: " + key);
        }
        return found->second;
    }

} // namespace instant

#endif // INSTANT_UTILITY_HPP
