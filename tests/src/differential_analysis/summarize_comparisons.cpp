#include <gtest/gtest.h>

#include "scran/differential_analysis/summarize_comparisons.hpp"

#include <numeric>

TEST(SummarizeComparisonsTest, Medians) {
    std::vector<double> stuff{0.2, 0.1, 0.3, 0.5, 0.4};

    auto copy = stuff;
    EXPECT_EQ(scran::differential_analysis::median(copy.data(), copy.size()), 0.3);

    copy = stuff;
    EXPECT_EQ(scran::differential_analysis::median(copy.data(), copy.size() - 1), 0.25);
}

class SummarizeComparisonsTest : public ::testing::TestWithParam<std::tuple<int> > {
protected:
    template<class Param>
    void assemble(Param& param) {
        ngroups = std::get<0>(param);
        output.resize(ngroups * 4 * ngenes);
        auto ptr = output.data();
        for (int g = 0; g < ngroups; ++g, ptr += 4 * ngenes) {
            ptrs.push_back(ptr);
        }
    }

    std::vector<double> output;
    std::vector<double*> ptrs;
    int ngroups, ngenes = 3;
};

/* For each group, the effect sizes start from `group_id * ngenes` for the
 * comparison to group 0 and increase consecutively until the comparison to
 * group `ngroups - 1`. This should make it relatively straightforward to
 * compute the various metrics exactly for testing purposes. We add a `gene`
 * multiplier to ensure that `summarize_comparisons` responds to it properly.
 */
std::vector<double> spawn_simple_values(int ngroups, int gene) {
    std::vector<double> output(ngroups * ngroups);
    auto start = output.begin();
    for (int g = 0; g < ngroups; ++g, start += ngroups) {
        std::iota(start, start + ngroups, g * gene);
    }
    return output;
}

TEST_P(SummarizeComparisonsTest, Basic) {
    assemble(GetParam());

    for (int gene = 0; gene < ngenes; ++gene) {
        auto values = spawn_simple_values(ngroups, gene);
        scran::differential_analysis::summarize_comparisons(ngroups, values.data(), gene, ngenes, ptrs);
    }

    for (int gene = 0; gene < ngenes; ++gene) {
        for (int g = 0; g < ngroups; ++g) {
            // Checking that the minimum is correct.
            EXPECT_FLOAT_EQ(ptrs[g][gene], g * gene + (g == 0));

            // Checking that the maximum is correct.
            EXPECT_FLOAT_EQ(ptrs[g][gene + 3 * ngenes], g * gene + ngroups - 1 - (g == ngroups - 1));

            // Checking that the mean is correct.
            EXPECT_FLOAT_EQ(ptrs[g][gene + ngenes], g * gene + ((ngroups - 1)*ngroups/2.0 - g)/(ngroups-1));
        }
    }
}

std::vector<double> spawn_missing_values(int ngroups, int gene, int lost) {
    std::vector<double> output(ngroups * ngroups);
    auto start = output.begin();
    for (int g = 0; g < ngroups; ++g, start += ngroups) {
        if (g == lost) {
            std::fill(start, start + ngroups, std::numeric_limits<double>::quiet_NaN());
        } else {
            std::iota(start, start + ngroups, g * gene);
            *(start + lost) = std::numeric_limits<double>::quiet_NaN();
        }
    }
    return output;
};

TEST_P(SummarizeComparisonsTest, Missing) {
    assemble(GetParam());

    for (int lost = 0; lost < ngroups; ++lost) {
        for (int gene = 0; gene < ngenes; ++gene) {
            auto values = spawn_missing_values(ngroups, gene, lost);
            scran::differential_analysis::summarize_comparisons(ngroups, values.data(), gene, ngenes, ptrs);
        }

        for (int gene = 0; gene < ngenes; ++ gene) {
            for (int g = 0; g < ngroups; ++g) {
                if (g == lost) {
                    for (int i = 0; i < 4; ++i) {
                        EXPECT_TRUE(std::isnan(ptrs[g][gene + i * ngenes]));
                    }
                    continue;
                }

                // Checking that the minimum is correct.
                auto curmin = ptrs[g][gene]; 
                if ((g==0 && lost==1) || (g==1 && lost==0)) {
                    EXPECT_FLOAT_EQ(curmin, g * gene + 2);
                } else if (g==0 || lost==0) {
                    EXPECT_FLOAT_EQ(curmin, g * gene + 1);
                } else {
                    EXPECT_FLOAT_EQ(curmin, g * gene);
                }

                // Checking that the maximum is correct.
                auto curmax = ptrs[g][gene + 3 * ngenes];
                if ((g == ngroups - 1 && lost == ngroups - 2) || (g == ngroups - 2 && lost == ngroups - 1)) {
                    EXPECT_FLOAT_EQ(curmax, g * gene + ngroups - 3);
                } else if (g == ngroups - 1 || lost == ngroups - 1) {
                    EXPECT_FLOAT_EQ(curmax, g * gene + ngroups - 2);
                } else {
                    EXPECT_FLOAT_EQ(curmax, g * gene + ngroups - 1);
                }

                // Checking that the mean is correct.
                auto curmean = ptrs[g][gene + ngenes];
                if (lost == g) {
                    EXPECT_FLOAT_EQ(curmean, g * gene + ((ngroups - 1)*ngroups/2.0 - g)/(ngroups-1));
                } else {
                    EXPECT_FLOAT_EQ(curmean, g * gene + ((ngroups - 1)*ngroups/2.0 - g - lost)/(ngroups-2));
                }
            }
        }
    }
}

INSTANTIATE_TEST_CASE_P(
    SummarizeComparisons,
    SummarizeComparisonsTest,
    ::testing::Combine(
        ::testing::Values(3, 5, 7) // number of groups
    )
);

