#include <gtest/gtest.h>

#include "../data/data.h"

#include "scran/quality_control/IsOutlier.hpp"

std::vector<double> even_values = { 
    0.761164335, 0.347582428, 0.430822695, 0.888530395, 0.627701241, 0.678912751,
    0.097218816, 0.692552865, 0.143479605, 0.049907948, 0.648174966, 0.848563147,
    0.472604294, 0.022525487, 0.738983761, 0.915699533, 0.577269375, 0.799325422,
    0.554883985, 0.009624974, 0.215816610
};

TEST(IsOutlier, BasicTests) {
    scran::IsOutlier is;
    auto isres = is.run(even_values.size(), even_values.data());
    EXPECT_EQ(std::vector<uint8_t>(even_values.size()), isres.outliers);

    auto lower = isres.thresholds.lower;
    EXPECT_EQ(lower.size(), 1);
    EXPECT_FLOAT_EQ(lower[0], -0.4103915108);

    auto upper = isres.thresholds.upper;
    EXPECT_EQ(upper.size(), 1);
    EXPECT_FLOAT_EQ(upper[0], 1.564930261);

    // Stripping it back down to check that filtering works.
    isres = is.set_nmads(1).run(even_values.size(), even_values.data());

    lower = isres.thresholds.lower;
    EXPECT_FLOAT_EQ(lower[0], 0.2480490797);

    upper = isres.thresholds.upper;
    EXPECT_FLOAT_EQ(upper[0], 0.9064896703);

    std::vector<uint8_t> ref(even_values.size());
    for (size_t i = 0; i < ref.size(); ++i) {
        ref[i] = even_values[i] < lower[0] || even_values[i] > upper[0];
    }
    EXPECT_EQ(ref, isres.outliers);

    // Checking for odd.
    scran::IsOutlier odd;
    auto oddres = odd.set_nmads(2).run(even_values.size() - 1, even_values.data());

    lower = oddres.thresholds.lower;
    EXPECT_EQ(lower.size(), 1);
    EXPECT_FLOAT_EQ(lower[0], 0.05614316);

    upper = oddres.thresholds.upper;
    EXPECT_EQ(upper.size(), 1);
    EXPECT_FLOAT_EQ(upper[0], 1.148827451);
}

TEST(IsOutlier, OneSidedTests) {
    scran::IsOutlier is;
    auto isres = is.set_nmads(0.5).set_upper(false).run(even_values.size(), even_values.data());

    auto upper = isres.thresholds.upper;
    EXPECT_TRUE(std::isinf(upper[0]) && upper[0] > 0);

    std::vector<uint8_t> ref(even_values.size());
    for (size_t i = 0; i < ref.size(); ++i) {
        ref[i] = even_values[i] < 0.4126592;
    }
    EXPECT_EQ(ref, isres.outliers);

    // Trying on the other side as well.
    isres = is.set_upper(true).set_lower(false).run(even_values.size(), even_values.data());
    auto lower = isres.thresholds.lower;
    EXPECT_TRUE(std::isinf(lower[0]) && lower[0] < 0);

    for (size_t i = 0; i < ref.size(); ++i) {
        ref[i] = even_values[i] > 0.7418795;
    }
    EXPECT_EQ(ref, isres.outliers);
}
     
TEST(IsOutlier, LogTests) {
    scran::IsOutlier is;
    auto isres = is.set_nmads(0.5).set_log(true).run(even_values.size(), even_values.data());

    auto lower = isres.thresholds.lower;
    EXPECT_FLOAT_EQ(lower[0], 0.4535230764);

    auto upper = isres.thresholds.upper;
    EXPECT_FLOAT_EQ(upper[0], 0.7347805407);

    std::vector<uint8_t> ref(even_values.size());
    for (size_t i = 0; i < ref.size(); ++i) {
        ref[i] = even_values[i] < lower[0] || even_values[i] > upper[0];
    }
    EXPECT_EQ(ref, isres.outliers);

    // Handles occasional zero values.
    auto copy1 = even_values;
    copy1.push_back(0.0000001);
    auto copy2 = even_values;
    copy2.push_back(0);

    isres = is.run(copy1.size(), copy1.data());
    auto lower1 = isres.thresholds.lower;
    auto upper1 = isres.thresholds.upper;
    isres = is.run(copy2.size(), copy2.data());
    auto lower2 = isres.thresholds.lower;
    auto upper2 = isres.thresholds.upper;
    EXPECT_EQ(lower1, lower2);
    EXPECT_EQ(upper1, upper2);

    // Does something sensible with loads of zeroes.
    std::vector<double> empty(even_values.size());
    isres = is.set_log(true).run(empty.size(), empty.data());
    EXPECT_EQ(isres.thresholds.lower[0], 0);
    EXPECT_EQ(isres.thresholds.upper[0], 0);

    isres = is.set_lower(false).set_upper(false).run(empty.size(), empty.data());
    upper = isres.thresholds.upper;
    EXPECT_TRUE(std::isinf(upper[0]) && upper[0] > 0);
    lower = isres.thresholds.lower;
    EXPECT_TRUE(std::isinf(lower[0]) && lower[0] < 0);
}

TEST(IsOutlier, BlockTests) {
    std::vector<int> block = {
        0, 1, 2, 3, 
        1, 0, 2, 3,
        1, 2, 0, 3,
        1, 2, 3, 0,
        0, 0, 0, 0, 
        1
    };
    scran::IsOutlier is;
    is.set_blocks(block.size(), block.data());
    auto isres = is.run(even_values.size(), even_values.data());

    // Checking the thresholds.
    auto lower = isres.thresholds.lower;
    EXPECT_EQ(lower.size(), 4);
    auto upper = isres.thresholds.upper;
    EXPECT_EQ(upper.size(), 4);
    auto outliers = isres.outliers;

    for (size_t i = 0; i < 4; ++i) {
        std::vector<double> copy;
        std::vector<uint8_t> discard;
        for (size_t j = 0; j < block.size(); ++j) {
            if (block[j] == i) {
                copy.push_back(even_values[j]);
                discard.push_back(outliers[j]);
            }
        }

        scran::IsOutlier is2;
        auto is2res = is2.run(copy.size(), copy.data());
        EXPECT_EQ(lower[i], is2res.thresholds.lower[0]);
        EXPECT_EQ(upper[i], is2res.thresholds.upper[0]);
        EXPECT_EQ(discard, is2res.outliers);
    }
}
