#include <gtest/gtest.h>

#include "../data/data.h"

#include "scran/qc/IsOutlier.hpp"

std::vector<double> even_values = { 
    0.761164335, 0.347582428, 0.430822695, 0.888530395, 0.627701241, 0.678912751,
    0.097218816, 0.692552865, 0.143479605, 0.049907948, 0.648174966, 0.848563147,
    0.472604294, 0.022525487, 0.738983761, 0.915699533, 0.577269375, 0.799325422,
    0.554883985, 0.009624974, 0.215816610
};

TEST(IsOutlier, BasicTests) {
    scran::IsOutlier is(even_values.size());
    is.find_outliers(even_values.data());
    EXPECT_EQ(is.get_outliers(), std::vector<int>(even_values.size()));

    auto lower = is.get_lower_thresholds();
    EXPECT_EQ(lower.size(), 1);
    EXPECT_FLOAT_EQ(lower[0], -0.4103915108);

    auto upper = is.get_upper_thresholds();
    EXPECT_EQ(upper.size(), 1);
    EXPECT_FLOAT_EQ(upper[0], 1.564930261);

    // Stripping it back down to check that filtering works.
    is.set_nmads(1).find_outliers(even_values.data());

    lower = is.get_lower_thresholds();
    EXPECT_FLOAT_EQ(lower[0], 0.2480490797);

    upper = is.get_upper_thresholds();
    EXPECT_FLOAT_EQ(upper[0], 0.9064896703);

    std::vector<int> ref(even_values.size());
    for (size_t i = 0; i < ref.size(); ++i) {
        ref[i] = even_values[i] < lower[0] || even_values[i] > upper[0];
    }
    EXPECT_EQ(ref, is.get_outliers());

    // Checking for odd.
    scran::IsOutlier odd(even_values.size() - 1);
    odd.set_nmads(2).find_outliers(even_values.data());

    lower = odd.get_lower_thresholds();
    EXPECT_EQ(lower.size(), 1);
    EXPECT_FLOAT_EQ(lower[0], 0.05614316);

    upper = odd.get_upper_thresholds();
    EXPECT_EQ(upper.size(), 1);
    EXPECT_FLOAT_EQ(upper[0], 1.148827451);
}

TEST(IsOutlier, OneSidedTests) {
    scran::IsOutlier is(even_values.size());
    is.set_nmads(0.5).set_upper(false).find_outliers(even_values.data());

    auto upper = is.get_upper_thresholds();
    EXPECT_TRUE(std::isinf(upper[0]) && upper[0] > 0);

    std::vector<int> ref(even_values.size());
    for (size_t i = 0; i < ref.size(); ++i) {
        ref[i] = even_values[i] < 0.4126592;
    }
    EXPECT_EQ(ref, is.get_outliers());

    // Trying on the other side as well.
    is.set_upper(true).set_lower(false).find_outliers(even_values.data());
    auto lower = is.get_lower_thresholds();
    EXPECT_TRUE(std::isinf(lower[0]) && lower[0] < 0);

    for (size_t i = 0; i < ref.size(); ++i) {
        ref[i] = even_values[i] > 0.7418795;
    }
    EXPECT_EQ(ref, is.get_outliers());
}
     
TEST(IsOutlier, LogTests) {
    scran::IsOutlier is(even_values.size());
    is.set_nmads(0.5).set_log(true).find_outliers(even_values.data());

    auto lower = is.get_lower_thresholds();
    EXPECT_FLOAT_EQ(lower[0], 0.4535230764);

    auto upper = is.get_upper_thresholds();
    EXPECT_FLOAT_EQ(upper[0], 0.7347805407);

    std::vector<int> ref(even_values.size());
    for (size_t i = 0; i < ref.size(); ++i) {
        ref[i] = even_values[i] < lower[0] || even_values[i] > upper[0];
    }
    EXPECT_EQ(ref, is.get_outliers());

    // Handles occasional zero values.
    auto copy1 = even_values;
    copy1.push_back(0.0000001);
    auto copy2 = even_values;
    copy2.push_back(0);

    is.find_outliers(copy1.data());
    auto lower1 = is.get_lower_thresholds();
    auto upper1 = is.get_upper_thresholds();
    is.find_outliers(copy2.data());
    auto lower2 = is.get_lower_thresholds();
    auto upper2 = is.get_upper_thresholds();
    EXPECT_EQ(lower1, lower2);
    EXPECT_EQ(upper1, upper2);

    // Does something sensible with loads of zeroes.
    std::vector<double> empty(even_values.size());
    is.set_log(true).find_outliers(empty.data());
    EXPECT_EQ(is.get_lower_thresholds()[0], 0);
    EXPECT_EQ(is.get_upper_thresholds()[0], 0);

    is.set_lower(false).set_upper(false).find_outliers(empty.data());
    upper = is.get_upper_thresholds();
    EXPECT_TRUE(std::isinf(upper[0]) && upper[0] > 0);
    lower = is.get_lower_thresholds();
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
    scran::IsOutlier is(even_values.size());
    is.set_blocks(block.data());
    is.find_outliers(even_values.data());

    // Checking the thresholds.
    auto lower = is.get_lower_thresholds();
    EXPECT_EQ(lower.size(), 4);
    auto upper = is.get_upper_thresholds();
    EXPECT_EQ(upper.size(), 4);
    auto outliers = is.get_outliers();

    for (size_t i = 0; i < 4; ++i) {
        std::vector<double> copy;
        std::vector<int> discard;
        for (size_t j = 0; j < block.size(); ++j) {
            if (block[j] == i) {
                copy.push_back(even_values[j]);
                discard.push_back(outliers[j]);
            }
        }

        scran::IsOutlier is2(copy.size());
        is2.find_outliers(copy.data());
        EXPECT_EQ(lower[i], is2.get_lower_thresholds()[0]);
        EXPECT_EQ(upper[i], is2.get_upper_thresholds()[0]);
        EXPECT_EQ(discard, is2.get_outliers());
    }
}
