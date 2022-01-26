#include <gtest/gtest.h>

#ifdef TEST_SCRAN_CUSTOM_PARALLEL
#include "../utils/custom_parallel.h"
#endif

#include "scran/differential_analysis/summarize_comparisons.hpp"
#include "../utils/compare_vectors.h"

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
        output.resize(ngroups * scran::differential_analysis::n_summaries * ngenes);

        auto ptr = output.data();
        ptrs.resize(scran::differential_analysis::n_summaries);
        for (int s = 0; s < ptrs.size(); ++s) {
            ptrs[s].clear();
            for (int g = 0; g < ngroups; ++g, ptr += ngenes) {
                ptrs[s].push_back(ptr);
            }
        }
    }

    std::vector<double> output;
    std::vector<std::vector<double*> > ptrs;
    int ngroups, ngenes = 11;
};

/* For each group, the effect sizes start from `group_id * gene` for the
 * comparison to group 0 and increase consecutively until the comparison to
 * group `ngroups - 1`. This should make it relatively straightforward to
 * compute the various metrics exactly for testing purposes. We add a `gene`
 * multiplier to ensure that `summarize_comparisons` responds to it properly.
 */
std::vector<double> spawn_simple_values(int ngroups, int ngenes) {
    std::vector<double> output(ngroups * ngroups * ngenes);
    auto start = output.begin();
    for (int gene = 0; gene < ngenes; ++gene) {
        for (int g = 0; g < ngroups; ++g, start += ngroups) {
            std::iota(start, start + ngroups, g * gene);
        }
    }
    return output;
}

TEST_P(SummarizeComparisonsTest, Basic) {
    assemble(GetParam());
    auto values = spawn_simple_values(ngroups, ngenes);
    scran::differential_analysis::summarize_comparisons(ngenes, ngroups, values.data(), ptrs);

    for (int gene = 0; gene < ngenes; ++gene) {
        for (int g = 0; g < ngroups; ++g) {
            // Checking that the minimum is correct.
            EXPECT_FLOAT_EQ(ptrs[0][g][gene], g * gene + (g == 0));

            // Checking that the mean is correct.
            EXPECT_FLOAT_EQ(ptrs[1][g][gene], g * gene + ((ngroups - 1)*ngroups/2.0 - g)/(ngroups-1));

            // Checking that the median is between the min and max.
            EXPECT_TRUE(ptrs[2][g][gene] >= ptrs[0][g][gene]);
            EXPECT_TRUE(ptrs[2][g][gene] <= ptrs[3][g][gene]);

            // Checking that the maximum is correct.
            EXPECT_FLOAT_EQ(ptrs[3][g][gene], g * gene + ngroups - 1 - (g == ngroups - 1));
        }
    }
}

std::vector<double> spawn_missing_values(int ngroups, int ngenes, int lost) {
    std::vector<double> output(ngroups * ngroups * ngenes);
    auto start = output.begin();
    for (int gene = 0; gene < ngenes; ++gene) {
        for (int g = 0; g < ngroups; ++g, start += ngroups) {
            if (g == lost) {
                std::fill(start, start + ngroups, std::numeric_limits<double>::quiet_NaN());
            } else {
                std::iota(start, start + ngroups, g * gene);
                *(start + lost) = std::numeric_limits<double>::quiet_NaN();
            }
        }
    }
    return output;
};

TEST_P(SummarizeComparisonsTest, Missing) {
    assemble(GetParam());

    for (int lost = 0; lost < ngroups; ++lost) {
        auto values = spawn_missing_values(ngroups, ngenes, lost);
        scran::differential_analysis::summarize_comparisons(ngenes, ngroups, values.data(), ptrs);

        for (int gene = 0; gene < ngenes; ++ gene) {
            for (int g = 0; g < ngroups; ++g) {
                if (g == lost) {
                    for (int i = 0; i < 3; ++i) {
                        EXPECT_TRUE(std::isnan(ptrs[i][g][gene]));
                    }
                    continue;
                }

                // Checking that the minimum is correct.
                double baseline = g * gene;
                if ((g==0 && lost==1) || (g==1 && lost==0)) {
                    baseline += 2;
                } else if (g==0 || lost==0) {
                    baseline += 1;
                }
                EXPECT_FLOAT_EQ(ptrs[0][g][gene], baseline);

                // Checking that the mean is correct.
                baseline = g * gene;
                if (lost == g) {
                    baseline += ((ngroups - 1)*ngroups/2.0 - g)/(ngroups-1);
                } else {
                    baseline += ((ngroups - 1)*ngroups/2.0 - g - lost)/(ngroups-2);
                }
                EXPECT_FLOAT_EQ(ptrs[1][g][gene], baseline);

                // Checking that the maximum is correct.
                baseline = g * gene + ngroups - 1;
                if ((g==ngroups - 1 && lost==ngroups - 2) || (g==ngroups - 2  && lost==ngroups - 1)) {
                    baseline -= 2;
                } else if (g==ngroups - 1 || lost==ngroups - 1) {
                    baseline -= 1;
                }
                EXPECT_FLOAT_EQ(ptrs[3][g][gene], baseline);
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

class ComputeMinRankTest : public ::testing::Test {
protected:
    void configure(int ngenes, int ngroups) {
        output.resize(ngroups * ngenes);
        auto ptr = output.data();
        for (int g = 0; g < ngroups; ++g, ptr += ngenes) {
            ptrs.push_back(ptr);
        }
    }

    std::vector<double*> ptrs;
    std::vector<double> output;
};

TEST_F(ComputeMinRankTest, Basic) {
    size_t ngenes = 4, ngroups = 3;
    std::vector<double> effects { 
        0, 1, 1, 1, 0, 1, 1, 1, 0,
        0, 2, 2, 2, 0, 2, 2, 2, 0,
        0, 3, 3, 3, 0, 3, 3, 3, 0,
        0, 4, 4, 4, 0, 4, 4, 4, 0
    };

    configure(ngenes, ngroups);
    scran::differential_analysis::compute_min_rank(ngenes, ngroups, effects.data(), ptrs);
    for (size_t i = 0; i < ngroups; ++i) {
        compare_vectors(std::vector<double>{4, 3, 2, 1}, ngenes, output.data() + i * ngenes); // reversed, for maximum effect sizes.
    }
}

TEST_F(ComputeMinRankTest, LessBasic) {
    size_t ngenes = 4, ngroups = 3;
    std::vector<double> effects { 
        0, 1, 2, 2, 0, 4, 1, 3, 0,
        0, 2, 4, 3, 0, 3, 2, 1, 0,
        0, 3, 1, 1, 0, 2, 3, 2, 0,
        0, 4, 3, 4, 0, 1, 4, 4, 0
     /* 1  1  1  2  2  2  3  3  3 => comparisons for each group */
    };
    for (auto& e : effects) { e *= -1; }

    configure(ngenes, ngroups);
    scran::differential_analysis::compute_min_rank(ngenes, ngroups, effects.data(), ptrs);
    compare_vectors(std::vector<double>{1, 2, 1, 3}, ngenes, output.data());
    compare_vectors(std::vector<double>{2, 3, 1, 1}, ngenes, output.data() + ngenes);
    compare_vectors(std::vector<double>{1, 1, 2, 4}, ngenes, output.data() + ngenes * 2);
}
