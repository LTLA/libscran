#ifndef COMPARE_ALMOST_EQUAL_H
#define COMPARE_ALMOST_EQUAL_H

#include <gtest/gtest.h>

template<class V>
void compare_almost_equal(const V& left, const V& right) {
    ASSERT_EQ(left.size(), right.size());
    for (size_t i = 0; i < left.size(); ++i) {
        EXPECT_FLOAT_EQ(left[i], right[i]);
    }
    return;
}

#endif
