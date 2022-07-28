#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "scran/differential_analysis/auc.hpp"

class AUCTest : public ::testing::Test {
protected:
    double slow_reference(const std::vector<double>& left, const std::vector<double>& right, double threshold = 0) const {
        double auc = 0;
        for (auto l : left) {
            size_t rx = 0;
            double collected = 0;
            while (rx < right.size() && right[rx] + threshold < l) {
                ++collected;
                ++rx;
            }

            double ties = 0;
            while (rx < right.size() && right[rx] + threshold == l) {
                ++ties;
                ++rx;
            }

            auc += collected + ties * 0.5;
        }

        return auc / (left.size() * right.size());
    }

    typedef scran::differential_analysis::PairedStore PairedStore;
    PairedStore input; 
    std::vector<int> num_zeros, totals;
    std::vector<double> output;

    void add_to_store(std::vector<double>& contents) {
        std::sort(contents.begin(), contents.end());
        size_t group = num_zeros.size();
        num_zeros.resize(group + 1);
        totals.push_back(contents.size());
        for (auto c : contents) {
            if (c) {
                input.push_back(std::make_pair(c, group));
            } else {
                ++num_zeros[group];
            }
        }
    }
};

TEST_F(AUCTest, Self) {
    std::vector<double> group1 { 0, -0.1, 1, 2.2, 3.5, 5 }; 
    add_to_store(group1);
    add_to_store(group1);

    double n = group1.size();
    EXPECT_FLOAT_EQ(slow_reference(group1, group1), 0.5); // checking that the default calculation is correct.

    output.resize(4);
    scran::differential_analysis::compute_pairwise_auc(input, num_zeros, totals, output.data(), true);

    EXPECT_FLOAT_EQ(output[1], 0.5);
    EXPECT_FLOAT_EQ(output[2], 0.5);

    // Trying again with 3 groups.
    add_to_store(group1);

    output.clear();
    output.resize(9);
    scran::differential_analysis::compute_pairwise_auc(input, num_zeros, totals, output.data(), true);

    EXPECT_FLOAT_EQ(output[0 + 1], 0.5); 
    EXPECT_FLOAT_EQ(output[0 + 2], 0.5); 
    EXPECT_FLOAT_EQ(output[3 + 0], 0.5); 
    EXPECT_FLOAT_EQ(output[3 + 2], 0.5); 
    EXPECT_FLOAT_EQ(output[6 + 0], 0.5); 
    EXPECT_FLOAT_EQ(output[6 + 1], 0.5); 
}

TEST_F(AUCTest, NoZeros) {
    std::vector<double> group1 { -0.1, 1, 1, 2.3, 4 };
    std::vector<double> group2 { 1, 5, 3.4, 5, -0.1, 5, -0.2, -5 };
    std::vector<double> group3 { -0.12, 4, -0.1, 5, 2, -0.1, 3, 5, 6.2, 1.2, 1.11 };

    add_to_store(group1);
    add_to_store(group2);
    add_to_store(group3);

    output.resize(9);
    scran::differential_analysis::compute_pairwise_auc(input, num_zeros, totals, output.data(), true);

    EXPECT_FLOAT_EQ(output[3 + 0], slow_reference(group2, group1)); 
    EXPECT_FLOAT_EQ(output[6 + 0], slow_reference(group3, group1)); 
    EXPECT_FLOAT_EQ(output[6 + 1], slow_reference(group3, group2)); 

    EXPECT_FLOAT_EQ(output[0 + 1], 1 - output[3 + 0]);
    EXPECT_FLOAT_EQ(output[0 + 2], 1 - output[6 + 0]);
    EXPECT_FLOAT_EQ(output[3 + 2], 1 - output[6 + 1]); 
}

TEST_F(AUCTest, Zeros) {
    std::vector<double> group1 { 0, -0.1, 0, 0, 2.3, 4, 0 };
    std::vector<double> group2 { 0, 1, 5, 0, 5, 0, 5, -0.2, -5 };
    std::vector<double> group3 { -0.12, 4, 0, 5, 2, 0, 3, 5, 0, 1.2, 1.11 };

    add_to_store(group1);
    add_to_store(group2);
    add_to_store(group3);

    output.resize(9);
    scran::differential_analysis::compute_pairwise_auc(input, num_zeros, totals, output.data(), true);

    EXPECT_FLOAT_EQ(output[3 + 0], slow_reference(group2, group1)); 
    EXPECT_FLOAT_EQ(output[6 + 0], slow_reference(group3, group1)); 
    EXPECT_FLOAT_EQ(output[6 + 1], slow_reference(group3, group2)); 
}

TEST_F(AUCTest, ThresholdSelf) {
    std::vector<double> group { -1, 0, 1, 4, 3, 2, 5, 6, 7, 9 };
    double threshold = 1;

    add_to_store(group);
    add_to_store(group);

    for (double threshold = 0.5; threshold <= 2; ++threshold) { 
        output.clear();
        output.resize(4);

        scran::differential_analysis::compute_pairwise_auc(input, num_zeros, totals, output.data(), threshold, true);
        EXPECT_FLOAT_EQ(output[2], slow_reference(group, group, threshold));
        EXPECT_FLOAT_EQ(output[1], output[2]);
    }

    // Consistent results with a threshold of zero.
    output.clear();
    output.resize(4);
    scran::differential_analysis::compute_pairwise_auc(input, num_zeros, totals, output.data(), 0, true);

    double n = group.size();
    EXPECT_FLOAT_EQ(output[2], 0.5);
    EXPECT_FLOAT_EQ(output[1], output[2]);
}

TEST_F(AUCTest, ThresholdNoZero) {
    // Use 0.5 increments so that we get some juicy ties after adding the threshold.
    std::vector<double> group1 { 0.5, -0.5, 3, 2, -1.5 };
    std::vector<double> group2 { -0.5, 1.5, 1.5, 1.5, 2.5, -0.5, -0.5 };
    std::vector<double> group3 { -0.5, 6, 2, -1.5, 0.5, 0.15, 1, 2, 5 };

    add_to_store(group1);
    add_to_store(group2);
    add_to_store(group3);

    for (double threshold = 0; threshold <= 2; threshold += 0.5) {
        output.clear();
        output.resize(9);
        scran::differential_analysis::compute_pairwise_auc(input, num_zeros, totals, output.data(), threshold, true);

        EXPECT_FLOAT_EQ(output[0 + 1], slow_reference(group1, group2, threshold)); 
        EXPECT_FLOAT_EQ(output[0 + 2], slow_reference(group1, group3, threshold)); 
        EXPECT_FLOAT_EQ(output[3 + 2], slow_reference(group2, group3, threshold)); 
        EXPECT_FLOAT_EQ(output[3 + 0], slow_reference(group2, group1, threshold)); 
        EXPECT_FLOAT_EQ(output[6 + 0], slow_reference(group3, group1, threshold)); 
        EXPECT_FLOAT_EQ(output[6 + 1], slow_reference(group3, group2, threshold)); 
    }
}

TEST_F(AUCTest, ThresholdZeros) {
    std::vector<double> group1 { 0, 0.5, -0.5, 3, 2, -1.5, 0 };
    std::vector<double> group2 { -0.5, 0, 1.5, 1.5, 0, 1.5, 2.5, 0, -0.5, -0.5 };
    std::vector<double> group3 { -0.5, 6, 2, 0, -1.5, 0.5, 0.15, 1, 2, 0, 5 };

    add_to_store(group1);
    add_to_store(group2);
    add_to_store(group3);

    for (double threshold = 0; threshold <= 0; threshold += 0.5) {
        output.clear();
        output.resize(9);
        scran::differential_analysis::compute_pairwise_auc(input, num_zeros, totals, output.data(), threshold, true);

        EXPECT_FLOAT_EQ(output[0 + 1], slow_reference(group1, group2, threshold)); 
        EXPECT_FLOAT_EQ(output[0 + 2], slow_reference(group1, group3, threshold)); 
        EXPECT_FLOAT_EQ(output[3 + 2], slow_reference(group2, group3, threshold)); 
        EXPECT_FLOAT_EQ(output[3 + 0], slow_reference(group2, group1, threshold)); 
        EXPECT_FLOAT_EQ(output[6 + 0], slow_reference(group3, group1, threshold)); 
        EXPECT_FLOAT_EQ(output[6 + 1], slow_reference(group3, group2, threshold)); 
    }
}
