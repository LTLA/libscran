#include <gtest/gtest.h>
#include "../utils/macros.h"

#include <cmath>
#include <vector>
#include "scran/differential_analysis/delta_detected.hpp"

TEST(DeltaDetected, Unblocked) {
    std::vector<int> detected{ 1, 2, 3, 4 };
    std::vector<int> group_sizes{ 10, 5, 12, 34 }; 

    std::vector<double> output(detected.size() * detected.size());
    scran::differential_analysis::compute_pairwise_delta_detected(detected.data(), group_sizes, detected.size(), 1, output.data());

    for (size_t g = 0; g < detected.size(); ++g) {
        for (size_t g2 = 0; g2 < detected.size(); ++g2) {
            double left = detected[g];
            double right = detected[g2];
            EXPECT_FLOAT_EQ(output[g * detected.size() + g2], left / group_sizes[g] - right / group_sizes[g2]);
        }
    }
}

TEST(DeltaDetected, ZeroGroups) {
    std::vector<int> detected{ 0, 2, 3, 4, 1 };
    std::vector<int> group_sizes{ 0, 5, 6, 7, 5 }; 

    std::vector<double> output(detected.size() * detected.size());
    scran::differential_analysis::compute_pairwise_delta_detected(detected.data(), group_sizes, detected.size(), 1, output.data());

    for (size_t g = 0; g < detected.size(); ++g) {
        for (size_t g2 = 0; g2 < detected.size(); ++g2) {
            if (g == g2) {
                continue;
            } 
            
            double x = output[g * detected.size() + g2];
            if (group_sizes[g] == 0 || group_sizes[g2] == 0) {
                EXPECT_TRUE(std::isnan(x));
            } else {
                double left = detected[g];
                double right = detected[g2];
                EXPECT_FLOAT_EQ(x, left / group_sizes[g] - right / group_sizes[g2]);
                double delta = detected[g] - detected[g2];
            }
        }
    }
}

TEST(DeltaDetected, Blocked) {
    int nblocks = 2, ngroups = 4;
    std::vector<int> detected{ 1, 2, 3, 4, 5, 6, 7, 8 };
    std::vector<int> group_sizes{ 10, 5, 12, 34, 15, 21, 31, 16 }; 

    std::vector<double> output(ngroups * ngroups);
    scran::differential_analysis::compute_pairwise_delta_detected(detected.data(), group_sizes, ngroups, nblocks, output.data());

    for (size_t g1 = 0; g1 < ngroups; ++g1) {
        int offset1 = g1 * nblocks;

        for (size_t g2 = 0; g2 < ngroups; ++g2) {
            int offset2 = g2 * nblocks;
            double totalnum = 0, totaldenom = 0;

            for (int b = 0; b < nblocks; ++b) {
                double left = static_cast<double>(detected[offset1 + b]) / group_sizes[offset1 + b];
                double right = static_cast<double>(detected[offset2 + b]) / group_sizes[offset2 + b];
                double d = left - right;
                double w = group_sizes[offset1 + b] * group_sizes[offset2 + b];
                totalnum += d * w;
                totaldenom += w;
            }

            EXPECT_FLOAT_EQ(output[g1 * ngroups + g2], totalnum/totaldenom);
        }
    }
}

TEST(DeltaDetected, BlockedZeroSize) {
    int nblocks = 2, ngroups = 4;
    std::vector<int> detected{ 0, 2, 0, 4, 0, 6, 0, 8 };
    std::vector<int> group_sizes{ 0, 5, 0, 34, 0, 23, 0, 6 }; 

    std::vector<double> output(ngroups * ngroups);
    scran::differential_analysis::compute_pairwise_delta_detected(detected.data(), group_sizes, ngroups, nblocks, output.data());

    // Effectively excising the first block.
    std::vector<double> output2(ngroups * ngroups);

    std::vector<int> subgroup_sizes;
    std::vector<int> sub_detected;
    for (size_t i = 0; i < detected.size(); ++i) {
        if (group_sizes[i] != 0) {
            sub_detected.push_back(detected[i]);
            subgroup_sizes.push_back(group_sizes[i]);
        }
    }

    scran::differential_analysis::compute_pairwise_delta_detected(sub_detected.data(), subgroup_sizes, ngroups, nblocks - 1, output2.data());

    EXPECT_EQ(output, output2);
}
