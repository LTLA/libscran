#ifndef COMPARE_ALMOST_EQUAL_H
#define COMPARE_ALMOST_EQUAL_H

#include <gtest/gtest.h>

template<class V>
void compare_almost_equal(const V& left, const V& right) {
    ASSERT_EQ(left.size(), right.size());

    for (size_t i = 0; i < left.size(); ++i) {
        if (left[i] == right[i]) {
            continue;
        }

        // Use relative tolerance to check that we get a similar float.
        // EXPECT_DOUBLE_EQ and EXPECT_FLOAT_EQ tend to be a bit too stringent
        // when numerical imprecision is expected. We define the relative difference
        // as |difference| / |mean| of the two values, and check that this is 
        // below some acceptable error, in this case 1e-8.
        //
        // It's worth nothing that I spent a long time using EXPECT_FLOAT_EQ
        // when I really should have been using EXPECT_DOUBLE_EQ; this would have
        // implicitly used a relative threshold at the 8th significant figure,
        // which is why it never failed from differences in numerical precision.
        double denom = std::abs(left[i] + right[i]) / 2;
        double threshold = denom * 1e-8;

        // This is effectively refactored to |difference| < threshold. However,
        // if threshold falls below the machine epsilon - usually because
        // |mean| is very small - we're effectively back to an exact equality
        // test, so we need to ensure that the threshold doesn't get too small.
        //
        // This compromises the stringency of the test at very-near-zero values,
        // but this could also be viewed as a feature, because we DON'T want to
        // fail the test if one value is zero and another is slightly non-zero
        // (e.g., when something should, in theory, perfectly cancel out).
        if (threshold < 1e-15) { 
            threshold = 1e-15;
        }
        if (std::abs(left[i] - right[i]) > threshold) {
            EXPECT_TRUE(false) << "mismatch at element " << i << " (expected " << left[i] << ", got " << right[i] << ")";
            return;
        }
    }

    return;
}

#endif
