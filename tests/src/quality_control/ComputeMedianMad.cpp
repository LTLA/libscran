#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "scran/quality_control/ComputeMedianMad.hpp"

std::vector<double> even_values = { 
    0.761164335, 0.347582428, 0.430822695, 0.888530395, 0.627701241, 0.678912751,
    0.097218816, 0.692552865, 0.143479605, 0.049907948, 0.648174966, 0.848563147,
    0.472604294, 0.022525487, 0.738983761, 0.915699533, 0.577269375, 0.799325422,
    0.554883985, 0.009624974, 0.215816610
};

TEST(ComputeMedianMad, BasicTests) {
    scran::ComputeMedianMad is;
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

TEST(ComputeMedianMad, EdgeTests) {
    scran::ComputeMedianMad is;

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

    empty.push_back(-std::numeric_limits<double>::infinity()); // add twice to keep it even.
    empty.push_back(-std::numeric_limits<double>::infinity());
    stats = is.run(empty.size(), empty.data());
    EXPECT_TRUE(std::isinf(stats.medians[0]));
    EXPECT_TRUE(stats.medians[0] < 0);
    EXPECT_EQ(stats.mads[0], 0);

    // Filtering works correctly in this case.
    {
        auto copy = even_values;
        auto ref = is.run(copy.size(), copy.data());

        copy.push_back(std::numeric_limits<double>::quiet_NaN());
        auto withnan = is.run(copy.size(), copy.data());

        EXPECT_EQ(ref.medians[0], withnan.medians[0]);
        EXPECT_EQ(ref.mads[0], withnan.mads[0]);
    }
}

TEST(ComputeMedianMad, MedianOnly) {
    scran::ComputeMedianMad is;
    is.set_median_only(true);
    auto stats = is.run(even_values.size(), even_values.data());

    EXPECT_FALSE(stats.log);
    EXPECT_EQ(stats.medians.size(), 1);
    EXPECT_FLOAT_EQ(stats.medians[0], 0.5772693750);
    EXPECT_TRUE(std::isnan(stats.mads[0]));
}

TEST(ComputeMedianMad, LogTests) {
    scran::ComputeMedianMad is;
    is.set_log(true);

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

TEST(ComputeMedianMad, BlockTests) {
    std::vector<int> block = {
        0, 1, 2, 3, 
        1, 0, 2, 3,
        1, 2, 0, 3,
        1, 2, 3, 0,
        0, 0, 0, 0, 
        1
    };

    scran::ComputeMedianMad is;
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

    // NULL blocked falls back to single-batch processing.
    {
        auto ref = is.run(even_values.size(), even_values.data());
        std::vector<double> buffer(even_values.size());
        auto isres_none = is.run_blocked(even_values.size(), static_cast<int*>(NULL), {}, even_values.data(), buffer.data());
        EXPECT_EQ(ref.medians, isres_none.medians);
        EXPECT_EQ(ref.mads, isres_none.mads);
    }
}
