#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "scran/differential_analysis/cohens_d.hpp"

TEST(CohensD, RawEdgeCases) {
    EXPECT_EQ(scran::differential_analysis::compute_cohens_d(1, 0, 0.5, 0), 2.0);
    EXPECT_EQ(scran::differential_analysis::compute_cohens_d(1, 0, 0.5, 1), 0.0);
    EXPECT_EQ(scran::differential_analysis::compute_cohens_d(1, 0, 0, 0), std::numeric_limits<double>::infinity());
    EXPECT_EQ(scran::differential_analysis::compute_cohens_d(0, 1, 0, 0), -std::numeric_limits<double>::infinity());
    EXPECT_EQ(scran::differential_analysis::compute_cohens_d(0, 0, 0, 0), 0);
    EXPECT_TRUE(std::isnan(scran::differential_analysis::compute_cohens_d(0, 0, std::numeric_limits<double>::quiet_NaN(), 0)));
}

TEST(CohensD, Denominator) {
    EXPECT_FLOAT_EQ(scran::differential_analysis::cohen_denominator(4.0, 0), std::sqrt(2.0));
    EXPECT_FLOAT_EQ(scran::differential_analysis::cohen_denominator(5.0, 3.0), 2.0);

    double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FLOAT_EQ(scran::differential_analysis::cohen_denominator(4.0, nan), 2.0);
    EXPECT_FLOAT_EQ(scran::differential_analysis::cohen_denominator(nan, 4.0), 2.0);
    EXPECT_TRUE(std::isnan(scran::differential_analysis::cohen_denominator(nan, nan)));
}

TEST(CohensD, Unblocked) {
    std::vector<double> means{0.1, 0.2, 0.3, 0.4};
    std::vector<double> variances{1.5, 2.3, 0.5, 1.2};
    std::vector<int> group_sizes{ 10, 5, 12, 34 }; // don't really matter.

    std::vector<double> output(means.size() * means.size());
    scran::differential_analysis::compute_pairwise_cohens_d(means.data(), variances.data(), group_sizes, means.size(), 1, 0, output.data());

    for (size_t g = 0; g < means.size(); ++g) {
        for (size_t g2 = 0; g2 < means.size(); ++g2) {
            EXPECT_FLOAT_EQ(output[g * means.size() + g2], (means[g] - means[g2]) / std::sqrt((variances[g] + variances[g2])/2.0));
        }
    }
}

TEST(CohensD, Thresholded) {
    std::vector<double> means{0.1, 0.2, 0.3, 0.4};
    std::vector<double> variances{1.5, 2.3, 0.5, 1.2};
    std::vector<int> group_sizes{ 10, 5, 12, 34 }; // don't really matter.

    std::vector<double> output(means.size() * means.size());
    scran::differential_analysis::compute_pairwise_cohens_d(means.data(), variances.data(), group_sizes, means.size(), 1, 1, output.data());

    for (size_t g = 0; g < means.size(); ++g) {
        for (size_t g2 = 0; g2 < means.size(); ++g2) {
            if (g != g2) {
                EXPECT_FLOAT_EQ(output[g * means.size() + g2], (means[g] - means[g2] - 1) / std::sqrt((variances[g] + variances[g2])/2.0));
            }
        }
    }
}

TEST(CohensD, MissingValues) {
    double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> means{ nan, 0.2, 0.3, 0.4, 0.1 };
    std::vector<double> variances{ nan, 2.3, nan, nan, 1.2 };
    std::vector<int> group_sizes{ 0, 5, 1, 1, 5 }; 

    std::vector<double> output(means.size() * means.size());
    scran::differential_analysis::compute_pairwise_cohens_d(means.data(), variances.data(), group_sizes, means.size(), 1, 0, output.data());

    for (size_t g = 0; g < means.size(); ++g) {
        for (size_t g2 = 0; g2 < means.size(); ++g2) {
            if (g == g2) {
                continue;
            } 
            
            double x = output[g * means.size() + g2];
            if (std::isnan(means[g]) || std::isnan(means[g2]) || (std::isnan(variances[g]) && std::isnan(variances[g2]))) {
                EXPECT_TRUE(std::isnan(x));
            } else {
                double delta = means[g] - means[g2];
                if (!std::isnan(variances[g]) && !std::isnan(variances[g2])) {
                    EXPECT_FLOAT_EQ(x, delta / std::sqrt((variances[g] + variances[g2])/2.0));
                } else if (!std::isnan(variances[g])) {
                    EXPECT_FLOAT_EQ(x, delta / std::sqrt(variances[g]));
                } else {
                    EXPECT_FLOAT_EQ(x, delta / std::sqrt(variances[g2]));
                }
            }
        }
    }
}

TEST(CohensD, Blocked) {
    int nblocks = 2, ngroups = 4;
    std::vector<double> means{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 };
    std::vector<double> variances{1.5, 2.3, 0.5, 1.2, 0.1, 1.2, 0.4, 0.5 };
    std::vector<int> group_sizes{ 10, 5, 12, 34, 15, 2, 3, 6 }; 

    std::vector<double> output(ngroups * ngroups);
    scran::differential_analysis::compute_pairwise_cohens_d(means.data(), variances.data(), group_sizes, ngroups, nblocks, 0, output.data());

    for (size_t g1 = 0; g1 < ngroups; ++g1) {
        int offset1 = g1 * nblocks;

        for (size_t g2 = 0; g2 < ngroups; ++g2) {
            int offset2 = g2 * nblocks;
            double totalnum = 0, totaldenom = 0;

            for (int b = 0; b < nblocks; ++b) {
                double d = (means[offset1 + b] - means[offset2 + b])/std::sqrt((variances[offset1 + b] + variances[offset2 + b])/2);
                double w = group_sizes[offset1 + b] * group_sizes[offset2 + b];
                totalnum += d * w;
                totaldenom += w;
            }

            EXPECT_FLOAT_EQ(output[g1 * ngroups + g2], totalnum/totaldenom);
        }
    }
}

TEST(CohensD, BlockedMissing) {
    double nan = std::numeric_limits<double>::quiet_NaN();
    int nblocks = 2, ngroups = 4;
    std::vector<double> means{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 };
    std::vector<double> variances{nan, 0.1, nan, 1.2, nan, 0.4, nan, 0.5 };
    std::vector<int> group_sizes{ 10, 5, 12, 34, 15, 2, 3, 6 }; 

    std::vector<double> output(ngroups * ngroups);
    scran::differential_analysis::compute_pairwise_cohens_d(means.data(), variances.data(), group_sizes, ngroups, nblocks, 0, output.data());

    // Effectively excising the first block.
    std::vector<double> output2(ngroups * ngroups);

    std::vector<int> subgroup_sizes;
    std::vector<double> sub_means, sub_variances;
    for (size_t i = 0; i < means.size(); ++i) {
        if (!std::isnan(variances[i])) {
            sub_means.push_back(means[i]);
            sub_variances.push_back(variances[i]);
            subgroup_sizes.push_back(group_sizes[i]);
        }
    }

    scran::differential_analysis::compute_pairwise_cohens_d(sub_means.data(), sub_variances.data(), subgroup_sizes, ngroups, nblocks - 1, 0, output2.data());

    EXPECT_EQ(output, output2);
}
