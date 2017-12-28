#include <gtest/gtest.h>

#include "np_io.hpp"

namespace instant {
    template<typename Iter1, typename Iter2>
    auto assert_eq_list(Iter1 first1, Iter1 last1, Iter2 first2, Iter2 last2) {
        ASSERT_EQ(std::distance(first1, last1), std::distance(first2, last2)) << "size is different";
        while(first1 != last1) {
            ASSERT_EQ(*first1, *first2) << *first1 << " and " << *first2 << " are different";
            ++first1;
            ++first2;
        }
    }

    template<typename List1, typename List2>
    auto assert_eq_list(List1 const& list1, List2 const& list2) {
        using std::begin;
        using std::end;
        assert_eq_list(begin(list1), end(list1), begin(list2), end(list2));
    }

    namespace {

        class NumpyIOTest : public ::testing::Test {};

        TEST_F(NumpyIOTest, load_test) {
            int dims_num;
            std::vector<int> shape;
            std::vector<float> data;
            std::tie(dims_num, shape, data) =
              instant::load_np_array("../data/input_3_5_64_64.txt");
            ASSERT_EQ(dims_num, 4);
            assert_eq_list(shape, std::array<int, 4>{{3, 5, 64, 64}});
        }

        TEST_F(NumpyIOTest, load_as_arr_test) {
            auto arr = instant::load_np_array_as_array("../data/input_3_5_64_64.txt");
            ASSERT_EQ(arr.dims().size(), 4);
            assert_eq_list(arr.dims(), std::array<int, 4>{{3, 5, 64, 64}});
        }
    } // namespace
} // namespace instant
