#include <gtest/gtest.h>

#include <instant/context.hpp>

namespace instant {
    namespace {

        class ContextTest : public ::testing::Test {};

        TEST_F(ContextTest, test_scoped_context) {
            std::cout << "available cpu count is "
                      << instant::get_available_cpu_count() << std::endl;
            ASSERT_TRUE(::instant::get_context().cpu_id() == 0);
            if(1 < instant::get_available_cpu_count()) {
                instant::scoped_context(1);
                ASSERT_TRUE(::instant::get_context().cpu_id() == 1);
            }
            ASSERT_TRUE(::instant::get_context().cpu_id() == 0);
        }

        TEST_F(ContextTest, test_scoped_context2) {
            std::cout << "available cpu count is "
                      << instant::get_available_cpu_count() << std::endl;
            ASSERT_TRUE(::instant::get_context().cpu_id() == 0);
            if(1 < instant::get_available_cpu_count()) {
                instant::scoped_context(1);
                ASSERT_TRUE(::instant::get_context().cpu_id() == 1);
                if(2 < instant::get_available_cpu_count()) {
                    instant::scoped_context(2);
                    ASSERT_TRUE(::instant::get_context().cpu_id() == 2);
                }
                ASSERT_TRUE(::instant::get_context().cpu_id() == 1);
            }
            ASSERT_TRUE(::instant::get_context().cpu_id() == 0);
        }

    } // namespace
} // namespace instant
