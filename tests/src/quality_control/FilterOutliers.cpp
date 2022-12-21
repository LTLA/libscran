#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "scran/quality_control/FilterOutliers.hpp"

TEST(FilterOutliers, Simple) {
    scran::ComputeMedianMad::Results stat;
    stat.medians.push_back(1);
    stat.mads.push_back(0.2);

    // Manual check.
    {
        scran::FilterOutliers filt;
        auto thresholds = filt.run(stat);
        EXPECT_DOUBLE_EQ(thresholds.lower[0], 0.4);
        EXPECT_DOUBLE_EQ(thresholds.upper[0], 1.6);
    }

    // Turns off on request.
    {
        scran::FilterOutliers filt;
        filt.lower = false;
        filt.upper = false;
        auto thresholds = filt.run(stat);

        EXPECT_TRUE(std::isinf(thresholds.lower[0]));
        EXPECT_TRUE(thresholds.lower[0] < 0);
        EXPECT_TRUE(std::isinf(thresholds.upper[0]));
        EXPECT_TRUE(thresholds.upper[0] > 0);
    }

    // Respects the minimum difference.
    {
        scran::FilterOutliers filt;
        filt.min_diff = 100;
        auto thresholds = filt.run(stat);
        EXPECT_DOUBLE_EQ(thresholds.lower[0], stat.medians[0] - 100);
        EXPECT_DOUBLE_EQ(thresholds.upper[0], stat.medians[0] + 100);
    }

    // Handles blocks correctly.
    {
        stat.medians.push_back(2);
        stat.mads.push_back(0.3);

        scran::FilterOutliers filt;
        auto thresholds = filt.run(stat);
        EXPECT_DOUBLE_EQ(thresholds.lower[0], 0.4);
        EXPECT_DOUBLE_EQ(thresholds.upper[0], 1.6);
        EXPECT_DOUBLE_EQ(thresholds.lower[1], 1.1);
        EXPECT_DOUBLE_EQ(thresholds.upper[1], 2.9);
    }
}

TEST(FilterOutliers, Logged) {
    scran::ComputeMedianMad::Results stat;
    stat.medians.push_back(1);
    stat.mads.push_back(0.2);
    stat.log = true;

    // Manual check.
    {
        scran::FilterOutliers filt;
        auto thresholds = filt.run(stat);
        EXPECT_DOUBLE_EQ(thresholds.lower[0], std::exp(0.4));
        EXPECT_DOUBLE_EQ(thresholds.upper[0], std::exp(1.6));
    }

    // Mostly zeros.
    {
        auto copy = stat;
        copy.medians[0] = -std::numeric_limits<double>::infinity();
        copy.mads[0] = 0;

        scran::FilterOutliers filt;
        auto thresholds = filt.run(copy);
        EXPECT_EQ(thresholds.lower[0], 0);
        EXPECT_EQ(thresholds.upper[0], 0);
    }
}

TEST(FilterOutliers, EdgeCases) {
    scran::ComputeMedianMad::Results stats;
    stats.medians.push_back(std::numeric_limits<double>::quiet_NaN());
    stats.mads.push_back(std::numeric_limits<double>::quiet_NaN());
    stats.medians.push_back(std::numeric_limits<double>::infinity());
    stats.mads.push_back(0);

    scran::FilterOutliers filt;
    auto thresholds = filt.run(stats);
    EXPECT_TRUE(std::isnan(thresholds.lower[0]));
    EXPECT_TRUE(std::isnan(thresholds.upper[0]));
    EXPECT_TRUE(std::isinf(thresholds.lower[1]));
    EXPECT_TRUE(thresholds.lower[1] > 0);
    EXPECT_TRUE(std::isinf(thresholds.upper[1]));
    EXPECT_TRUE(thresholds.upper[1] > 0);
}

