#include <gtest/gtest.h>

#include "scran/differential_analysis/summarize_comparisons.hpp"

#include <numeric>

TEST(SummarizeComparisonsTest, Quantile) {
    std::vector<double> stuff{0.1, 0.2, 0.3, 0.4, 0.5};

    // Medians work.
    EXPECT_EQ(scran::differential_analysis::quantile(stuff.begin(), stuff.size(), 1, 2), 0.3);
    EXPECT_EQ(scran::differential_analysis::quantile(stuff.begin(), stuff.size() - 1, 1, 2), 0.25);

    // Extremes work.
    EXPECT_EQ(scran::differential_analysis::quantile(stuff.begin(), stuff.size(), 0, 2), 0.1);
    EXPECT_EQ(scran::differential_analysis::quantile(stuff.begin(), stuff.size(), 2, 2), 0.5);

    // Tricky quantiles work.
    EXPECT_EQ(scran::differential_analysis::quantile(stuff.begin(), stuff.size(), 1, 4), 0.2);
    EXPECT_EQ(scran::differential_analysis::quantile(stuff.begin(), stuff.size(), 3, 4), 0.4);

    EXPECT_FLOAT_EQ(scran::differential_analysis::quantile(stuff.begin(), stuff.size(), 1, 10), 0.14);
    EXPECT_FLOAT_EQ(scran::differential_analysis::quantile(stuff.begin(), stuff.size(), 9, 10), 0.46);

    // Trying with more irregular spacings.
    std::vector<double> stuff1{0.01, 0.1, 0.2, 0.4, 0.41, 0.5};

    EXPECT_FLOAT_EQ(scran::differential_analysis::quantile(stuff1.begin(), stuff1.size(), 65, 100), 0.4025);
    EXPECT_FLOAT_EQ(scran::differential_analysis::quantile(stuff1.begin(), stuff1.size(), 33, 100), 0.165);

    // Trying with duplicates.
    std::vector<double> stuff2{0, 0, 0, 1, 1, 1, 2, 2};

    EXPECT_FLOAT_EQ(scran::differential_analysis::quantile(stuff2.begin(), stuff2.size(), 1, 10), 0);
    EXPECT_FLOAT_EQ(scran::differential_analysis::quantile(stuff2.begin(), stuff2.size(), 5, 10), 1);
    EXPECT_FLOAT_EQ(scran::differential_analysis::quantile(stuff2.begin(), stuff2.size(), 9, 10), 2);
}

class SummarizeComparisonsTest : public ::testing::TestWithParam<std::tuple<int> > {
protected:
    template<class Param>
    void assemble(Param& param) {
        ngroups = std::get<0>(param);
        output.resize(ngroups * 4 * ngenes);
        ptrs.resize(ngroups);

        auto ptr = output.data();
        for (int g = 0; g < ngroups; ++g) {
            for (int x = 0; x < 4; ++x, ptr += ngenes) {
                ptrs[g].push_back(ptr);
            }
        }
    }

    std::vector<double> output;
    std::vector<std::vector<double*> > ptrs;
    int ngroups, ngenes = 3;
};

/* For each group, the effect sizes start from `group_id * ngenes` for the
 * comparison to group 0 and increase consecutively until the comparison to
 * group `ngroups - 1`. This should make it relatively straightforward to
 * compute the various metrics exactly for testing purposes.
 */
struct SpawnSimpleValues {
    SpawnSimpleValues(int ng) : ngroups(ng) {}
    void operator()(int gene, std::vector<double>& buffer) {
        auto start = buffer.begin();
        for (int g = 0; g < ngroups; ++g, start += ngroups) {
            std::iota(start, start + ngroups, g * gene);
        }
    }
    int ngroups;
};

TEST_P(SummarizeComparisonsTest, Basic) {
    assemble(GetParam());
    SpawnSimpleValues src(ngroups);
    scran::differential_analysis::summarize_comparisons(ngenes, ngroups, src, ptrs);

    for (int gene = 0; gene < ngenes; ++ gene) {
        for (int g = 0; g < ngroups; ++g) {
            // Checking that the minimum is correct.
            EXPECT_FLOAT_EQ(ptrs[g][0][gene], g * gene + (g == 0));

            // Checking that the maximum is correct.
            EXPECT_FLOAT_EQ(ptrs[g][3][gene], g * gene + ngroups - 1 - (g == ngroups - 1));

            // Checking that the mean is correct.
            EXPECT_FLOAT_EQ(ptrs[g][1][gene], g * gene + ((ngroups - 1)*ngroups/2.0 - g)/(ngroups-1));
        }
    }
}

struct SpawnMissingValues {
    SpawnMissingValues(int ng, int l) : ngroups(ng), lost(l) {}
    void operator()(int gene, std::vector<double>& buffer) {
        auto start = buffer.begin();
        for (int g = 0; g < ngroups; ++g, start += ngroups) {
            if (g == lost) {
                std::fill(start, start + ngroups, std::numeric_limits<double>::quiet_NaN());
            } else {
                std::iota(start, start + ngroups, g * gene);
                start[lost] = std::numeric_limits<double>::quiet_NaN();
            }
        }
    }
    int ngroups, lost;
};

TEST_P(SummarizeComparisonsTest, Missing) {
    assemble(GetParam());

    for (int lost = 0; lost < ngroups; ++lost) {
        SpawnMissingValues src(ngroups, lost);
        scran::differential_analysis::summarize_comparisons(ngenes, ngroups, src, ptrs);

        for (int gene = 0; gene < ngenes; ++ gene) {
            for (int g = 0; g < ngroups; ++g) {
                if (g == lost) {
                    for (int i = 0; i < 4; ++i) {
                        EXPECT_TRUE(std::isnan(ptrs[g][i][gene]));
                    }
                    continue;
                }

                // Checking that the minimum is correct.
                if ((g==0 && lost==1) || (g==1 && lost==0)) {
                    EXPECT_FLOAT_EQ(ptrs[g][0][gene], g * gene + 2);
                } else if (g==0 || lost==0) {
                    EXPECT_FLOAT_EQ(ptrs[g][0][gene], g * gene + 1);
                } else {
                    EXPECT_FLOAT_EQ(ptrs[g][0][gene], g * gene);
                }

                // Checking that the maximum is correct.
                if ((g == ngroups - 1 && lost == ngroups - 2) || (g == ngroups - 2 && lost == ngroups - 1)) {
                    EXPECT_FLOAT_EQ(ptrs[g][3][gene], g * gene + ngroups - 3);
                } else if (g == ngroups - 1 || lost == ngroups - 1) {
                    EXPECT_FLOAT_EQ(ptrs[g][3][gene], g * gene + ngroups - 2);
                } else {
                    EXPECT_FLOAT_EQ(ptrs[g][3][gene], g * gene + ngroups - 1);
                }

                // Checking that the mean is correct.
                if (lost == g) {
                    EXPECT_FLOAT_EQ(ptrs[g][1][gene], g * gene + ((ngroups - 1)*ngroups/2.0 - g)/(ngroups-1));
                } else {
                    EXPECT_FLOAT_EQ(ptrs[g][1][gene], g * gene + ((ngroups - 1)*ngroups/2.0 - g - lost)/(ngroups-2));
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

