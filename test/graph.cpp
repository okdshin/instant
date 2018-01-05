#include <gtest/gtest.h>

#include <instant/graph.hpp>

#include "./common.hpp"

namespace instant {

    namespace {

        class GraphTest : public ::testing::Test {};

        TEST_F(GraphTest, construct_test) {
            node n(op_type_t::max_pool, {"input"}, {"output"},
                 {{"strides", std::vector{2, 2}},
                  {"pads", std::vector{0, 0}},
                  {"ksize", std::vector{3, 3}}});
        }

        TEST_F(GraphTest, attribute_access_test) {
            node n(op_type_t::max_pool, {"input"}, {"output"},
                 {{"strides", std::vector{2, 2}},
                  {"pads", std::vector{0, 0}},
                  {"ksize", std::vector{3, 3}}});
            auto const& strides = attribute_ints(n, "strides");
            assert_eq_list(strides, std::vector{2, 2});
        }

    } // namespace

} // namespace instant
