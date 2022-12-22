#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "scran/quality_control/ChooseOutlierFilters.hpp"

TEST(ChooseOutlierFilters, Simple) {
    scran::ComputeMedianMad::Results stat;
    stat.medians.push_back(1);
    stat.mads.push_back(0.2);

    // Manual check.
    {
        scran::ChooseOutlierFilters filt;
        auto thresholds = filt.run(stat);
        EXPECT_DOUBLE_EQ(thresholds.lower[0], 0.4);
        EXPECT_DOUBLE_EQ(thresholds.upper[0], 1.6);
    }

    // Turns off on request.
    {
        scran::ChooseOutlierFilters filt;
        filt.set_lower(false);
        filt.set_upper(false);
        auto thresholds = filt.run(stat);
        EXPECT_TRUE(thresholds.lower.empty());
        EXPECT_TRUE(thresholds.upper.empty());
    }

    // Respects the minimum difference.
    {
        scran::ChooseOutlierFilters filt;
        filt.set_min_diff(100);
        auto thresholds = filt.run(stat);
        EXPECT_DOUBLE_EQ(thresholds.lower[0], stat.medians[0] - 100);
        EXPECT_DOUBLE_EQ(thresholds.upper[0], stat.medians[0] + 100);
    }

    // Handles blocks correctly.
    {
        stat.medians.push_back(2);
        stat.mads.push_back(0.3);

        scran::ChooseOutlierFilters filt;
        auto thresholds = filt.run(stat);
        EXPECT_DOUBLE_EQ(thresholds.lower[0], 0.4);
        EXPECT_DOUBLE_EQ(thresholds.upper[0], 1.6);
        EXPECT_DOUBLE_EQ(thresholds.lower[1], 1.1);
        EXPECT_DOUBLE_EQ(thresholds.upper[1], 2.9);
    }
}

TEST(ChooseOutlierFilters, Logged) {
    scran::ComputeMedianMad::Results stat;
    stat.medians.push_back(1);
    stat.mads.push_back(0.2);
    stat.log = true;

    // Manual check.
    {
        scran::ChooseOutlierFilters filt;
        auto thresholds = filt.run(stat);
        EXPECT_DOUBLE_EQ(thresholds.lower[0], std::exp(0.4));
        EXPECT_DOUBLE_EQ(thresholds.upper[0], std::exp(1.6));
    }

    // Mostly zeros.
    {
        auto copy = stat;
        copy.medians[0] = -std::numeric_limits<double>::infinity();
        copy.mads[0] = 0;

        scran::ChooseOutlierFilters filt;
        auto thresholds = filt.run(copy);
        EXPECT_EQ(thresholds.lower[0], 0);
        EXPECT_EQ(thresholds.upper[0], 0);
    }
}

TEST(ChooseOutlierFilters, EdgeCases) {
    scran::ComputeMedianMad::Results stats;
    stats.medians.push_back(std::numeric_limits<double>::quiet_NaN());
    stats.mads.push_back(std::numeric_limits<double>::quiet_NaN());
    stats.medians.push_back(std::numeric_limits<double>::infinity());
    stats.mads.push_back(0);

    scran::ChooseOutlierFilters filt;
    auto thresholds = filt.run(stats);
    EXPECT_TRUE(std::isnan(thresholds.lower[0]));
    EXPECT_TRUE(std::isnan(thresholds.upper[0]));
    EXPECT_TRUE(std::isinf(thresholds.lower[1]));
    EXPECT_TRUE(thresholds.lower[1] > 0);
    EXPECT_TRUE(std::isinf(thresholds.upper[1]));
    EXPECT_TRUE(thresholds.upper[1] > 0);

    // Checking that the filter handles NaNs correctly,
    // both in the values and in the thresholds.
    {
        std::vector<double> metrics { std::numeric_limits<double>::quiet_NaN(), 0, 1, std::numeric_limits<double>::quiet_NaN() };
        std::vector<int> block { 0, 0, 1, 1 };

        auto out = thresholds.filter_blocked(block.size(), block.data(), metrics.data());
        EXPECT_FALSE(out[0]); // both threshold and value are NaN.
        EXPECT_FALSE(out[1]); // comparison to NaN threshold.
        EXPECT_TRUE(out[2]); // finite comparison to +Inf lower threshold.
        EXPECT_FALSE(out[3]); // comparison to NaN value.
    }
}

TEST(ChooseOutlierFilters, Filters) {
    scran::ComputeMedianMad::Results stat;
    stat.medians.push_back(1);
    stat.mads.push_back(0.2);

    // Unblocked.
    {
        scran::ChooseOutlierFilters filt;

        std::vector<double> metrics { 0.1, 0.2, 0.3, 0.5, 1, 1.5, 1.7, 1.8, 1.9 };
        {
            auto thresholds = filt.run(stat);
            auto out = thresholds.filter(metrics.size(), metrics.data());
            std::vector<uint8_t> expected { 1, 1, 1, 0, 0, 0, 1, 1, 1 };
            EXPECT_EQ(out, expected);
        }

        filt.set_lower(false);
        {
            auto thresholds = filt.run(stat);
            auto out = thresholds.filter(metrics.size(), metrics.data());
            std::vector<uint8_t> expected { 0, 0, 0, 0, 0, 0, 1, 1, 1 };
            EXPECT_EQ(out, expected);
        }

        filt.set_lower(true);
        filt.set_upper(false);
        {
            auto thresholds = filt.run(stat);
            auto out = thresholds.filter(metrics.size(), metrics.data());
            std::vector<uint8_t> expected { 1, 1, 1, 0, 0, 0, 0, 0, 0 };
            EXPECT_EQ(out, expected);
        }
    }

    // Blocked.
    stat.medians.push_back(2);
    stat.mads.push_back(0.5);
    {
        std::vector<int> blocks { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1 };
        std::vector<double> metrics { 0.1, 0.9, 1, 1.1, 1.9, 2, 0.1, 2.1, 3.9, 1.9 };
        EXPECT_EQ(blocks.size(), metrics.size());

        scran::ChooseOutlierFilters filt;
        auto thresholds = filt.run(stat);
        EXPECT_ANY_THROW(thresholds.filter(metrics.size(), metrics.data()));

        auto out = thresholds.filter_blocked(blocks.size(), blocks.data(), metrics.data());
        std::vector<uint8_t> expected { 1, 0, 0, 0, 1, 0, 1, 0, 1, 0 };
        EXPECT_EQ(out, expected);
    }
}
