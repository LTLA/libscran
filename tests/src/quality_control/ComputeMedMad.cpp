#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "../data/data.h"

#include "scran/quality_control/ComputeMedMad.hpp"

std::vector<double> even_values = { 
    0.761164335, 0.347582428, 0.430822695, 0.888530395, 0.627701241, 0.678912751,
    0.097218816, 0.692552865, 0.143479605, 0.049907948, 0.648174966, 0.848563147,
    0.472604294, 0.022525487, 0.738983761, 0.915699533, 0.577269375, 0.799325422,
    0.554883985, 0.009624974, 0.215816610
};

TEST(ComputeMedMad, BasicTests) {
    scran::ComputeMedMad is;
    auto stats = is.run(even_values.size(), even_values.data());

    EXPECT_FALSE(stats.log);
    EXPECT_EQ(stats.medians.size(), 1);
    EXPECT_FLOAT_EQ(stats.medians[0], 0.5772693750);

    EXPECT_EQ(stats.mads.size(), 1);
    EXPECT_FLOAT_EQ(stats.mads[0], 0.3292202953);

    // Checking for odd.
    auto stats2 = is.run(even_values.size() - 1, even_values.data());

    EXPECT_EQ(stats2.medians.size(), 1);
    EXPECT_FLOAT_EQ(stats2.medians[0], 0.6024853080);

    EXPECT_EQ(stats2.mads.size(), 1);
    EXPECT_FLOAT_EQ(stats2.mads[0], 0.2731710715);
}

TEST(ComputeMedMad, EdgeTests) {
    scran::ComputeMedMad is;

    std::vector<double> empty;
    auto stats = is.run(empty.size(), empty.data());
    EXPECT_TRUE(std::isnan(stats.medians[0]));
    EXPECT_TRUE(std::isnan(stats.mads[0]));

    empty.push_back(std::numeric_limits<double>::infinity());
    stats = is.run(empty.size(), empty.data());
    EXPECT_TRUE(std::isinf(stats.medians[0]));
    EXPECT_TRUE(stats.medians[0] > 0);
    EXPECT_EQ(stats.mads[0], 0);
    
    empty.push_back(-std::numeric_limits<double>::infinity());
    stats = is.run(empty.size(), empty.data());
    EXPECT_TRUE(std::isnan(stats.medians[0]));
    EXPECT_TRUE(std::isnan(stats.mads[0]));

    empty.push_back(-std::numeric_limits<double>::infinity());
    stats = is.run(empty.size(), empty.data());
    EXPECT_TRUE(std::isinf(stats.medians[0]));
    EXPECT_TRUE(stats.medians[0] < 0);
    EXPECT_EQ(stats.mads[0], 0);

    {
        auto copy = even_values;
        auto ref = is.run(copy.size(), copy.data());

        copy.push_back(std::numeric_limits<double>::quiet_NaN());
        auto withnan = is.run(copy.size(), copy.data());

        EXPECT_EQ(ref.medians[0], withnan.medians[0]);
        EXPECT_EQ(ref.mads[0], withnan.mads[0]);
    }
}

TEST(ComputeMedMad, LogTests) {
    scran::ComputeMedMad is;
    is.log = true;

    auto isres = is.run(even_values.size(), even_values.data());
    EXPECT_TRUE(isres.log);
    EXPECT_FLOAT_EQ(isres.medians[0], -0.5494462670);
    EXPECT_FLOAT_EQ(isres.mads[0], 0.4825257172);

    // Handles occasional zero values.
    {
        auto copy1 = even_values;
        copy1.push_back(0.0000001);
        auto copy2 = even_values;
        copy2.push_back(0);

        auto isres1 = is.run(copy1.size(), copy1.data());
        auto isres2 = is.run(copy2.size(), copy2.data());
        EXPECT_EQ(isres1.medians[0], isres2.medians[0]);
        EXPECT_EQ(isres1.mads[0], isres2.mads[0]);
    }

    // Does something sensible with loads of zeroes.
    {
        std::vector<double> empty(even_values.size());
        auto isres = is.run(empty.size(), empty.data());
        EXPECT_TRUE(std::isinf(isres.medians[0]));
        EXPECT_TRUE(isres.medians[0] < 0);
        EXPECT_EQ(isres.mads[0], 0);
    }
}

TEST(ComputeMedMad, BlockTests) {
    std::vector<int> block = {
        0, 1, 2, 3, 
        1, 0, 2, 3,
        1, 2, 0, 3,
        1, 2, 3, 0,
        0, 0, 0, 0, 
        1
    };

    scran::ComputeMedMad is;
    auto isres = is.run_blocked(even_values.size(), block.data(), even_values.data());

    EXPECT_EQ(isres.medians.size(), 4);
    EXPECT_EQ(isres.mads.size(), 4);

    for (size_t i = 0; i < 4; ++i) {
        std::vector<double> copy;
        for (size_t j = 0; j < block.size(); ++j) {
            if (block[j] == i) {
                copy.push_back(even_values[j]);
            }
        }

        auto is2res = is.run(copy.size(), copy.data());
        EXPECT_EQ(isres.medians[i], is2res.medians[0]);
        EXPECT_EQ(isres.mads[i], is2res.mads[0]);
    }

    // NULL blocked is no-op.
    {
        auto isres_none = is.run_blocked(even_values.size(), static_cast<int*>(NULL), even_values.data());
        auto ref = is.run(even_values.size(), even_values.data());
        EXPECT_EQ(ref.medians, isres_none.medians);
        EXPECT_EQ(ref.mads, isres_none.mads);
    }
}

TEST(ComputeMedMad, OutlierFilterSimple) {
    scran::ComputeMedMad is;
    auto stat = is.run(even_values.size(), even_values.data());

    // Manual check.
    {
        scran::ComputeMedMad::FilterOutliers filt;
        auto thresholds = filt.run(stat);
        EXPECT_DOUBLE_EQ(thresholds.lower[0], stat.medians[0] - stat.mads[0] * 3);
        EXPECT_DOUBLE_EQ(thresholds.upper[0], stat.medians[0] + stat.mads[0] * 3);
    }

    // Turns off on request.
    {
        scran::ComputeMedMad::FilterOutliers filt;
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
        scran::ComputeMedMad::FilterOutliers filt;
        filt.min_diff = 100;
        auto thresholds = filt.run(stat);
        EXPECT_DOUBLE_EQ(thresholds.lower[0], stat.medians[0] - 100);
        EXPECT_DOUBLE_EQ(thresholds.upper[0], stat.medians[0] + 100);
    }

    // Handles blocks correctly.
    {
        scran::ComputeMedMad::Results stats;
        stats.medians.push_back(1);
        stats.mads.push_back(0.2);
        stats.medians.push_back(2);
        stats.mads.push_back(0.3);

        scran::ComputeMedMad::FilterOutliers filt;
        auto thresholds = filt.run(stats);
        EXPECT_DOUBLE_EQ(thresholds.lower[0], 0.4);
        EXPECT_DOUBLE_EQ(thresholds.upper[0], 1.6);
        EXPECT_DOUBLE_EQ(thresholds.lower[1], 1.1);
        EXPECT_DOUBLE_EQ(thresholds.upper[1], 2.9);
    }
}

TEST(ComputeMedMad, OutlierFilterLogged) {
    scran::ComputeMedMad is;
    is.log = true;
    auto stat = is.run(even_values.size(), even_values.data());

    // Manual check.
    {
        scran::ComputeMedMad::FilterOutliers filt;
        auto thresholds = filt.run(stat);
        EXPECT_DOUBLE_EQ(thresholds.lower[0], std::exp(stat.medians[0] - stat.mads[0] * 3));
        EXPECT_DOUBLE_EQ(thresholds.upper[0], std::exp(stat.medians[0] + stat.mads[0] * 3));
    }

    // Mostly zeros.
    {
        auto copy = stat;
        copy.medians[0] = -std::numeric_limits<double>::infinity();
        copy.mads[0] = 0;

        scran::ComputeMedMad::FilterOutliers filt;
        auto thresholds = filt.run(copy);
        EXPECT_EQ(thresholds.lower[0], 0);
        EXPECT_EQ(thresholds.upper[0], 0);
    }
}

TEST(ComputeMedMad, OutlierFilterEdgeCases) {
    scran::ComputeMedMad::Results stats;
    stats.medians.push_back(std::numeric_limits<double>::quiet_NaN());
    stats.mads.push_back(std::numeric_limits<double>::quiet_NaN());
    stats.medians.push_back(std::numeric_limits<double>::infinity());
    stats.mads.push_back(0);

    scran::ComputeMedMad::FilterOutliers filt;
    auto thresholds = filt.run(stats);
    EXPECT_TRUE(std::isnan(thresholds.lower[0]));
    EXPECT_TRUE(std::isnan(thresholds.upper[0]));
    EXPECT_TRUE(std::isinf(thresholds.lower[1]));
    EXPECT_TRUE(thresholds.lower[1] > 0);
    EXPECT_TRUE(std::isinf(thresholds.upper[1]));
    EXPECT_TRUE(thresholds.upper[1] > 0);
}
