#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "scran/differential_analysis/SummarizeEffects.hpp"
#include "scran/differential_analysis/summarize_comparisons.hpp"
#include "../utils/compare_vectors.h"

#include <numeric>
#include <random>

TEST(SummarizeComparisonsTest, Medians) {
    std::vector<double> stuff{0.2, 0.1, 0.3, 0.5, 0.4};

    auto copy = stuff;
    EXPECT_EQ(scran::differential_analysis::median(copy.data(), copy.size()), 0.3);

    copy = stuff;
    EXPECT_EQ(scran::differential_analysis::median(copy.data(), copy.size() - 1), 0.25);
}

/*********************************************/

class SummarizeComparisonsTest : public ::testing::TestWithParam<std::tuple<int, int> > {
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

    int ngroups, ngenes = 11;
    std::vector<double> output;
    std::vector<std::vector<double*> > ptrs;
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
    auto threads = std::get<1>(GetParam());

    scran::differential_analysis::summarize_comparisons(ngenes, ngroups, values.data(), ptrs, threads);

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

    // Checking the serial version for consistency.
    if (threads > 1) {
        auto parallelized = output;
        std::fill(output.begin(), output.end(), 0);
        scran::differential_analysis::summarize_comparisons(ngenes, ngroups, values.data(), ptrs, 1); 
        EXPECT_EQ(parallelized, output);
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
    auto threads = std::get<1>(GetParam());

    for (int lost = 0; lost < ngroups; ++lost) {
        auto values = spawn_missing_values(ngroups, ngenes, lost);
        scran::differential_analysis::summarize_comparisons(ngenes, ngroups, values.data(), ptrs, threads);

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

INSTANTIATE_TEST_SUITE_P(
    SummarizeComparisons,
    SummarizeComparisonsTest,
    ::testing::Combine(
        ::testing::Values(3, 5, 7), // number of groups
        ::testing::Values(1, 3) // number of threads
    )
);

/*********************************************/

class ComputeMinRankTest : public ::testing::TestWithParam<int> {
protected:
    void configure(int ngenes, int ngroups) {
        output.resize(ngroups * ngenes);
        auto ptr = output.data();
        for (int g = 0; g < ngroups; ++g, ptr += ngenes) {
            ptrs.push_back(ptr);
        }
    }

    std::vector<double> output;
    std::vector<double*> ptrs;
};

TEST_P(ComputeMinRankTest, Basic) {
    size_t ngenes = 4, ngroups = 3;
    std::vector<double> effects { 
        0, 1, 1, 1, 0, 1, 1, 1, 0,
        0, 2, 2, 2, 0, 2, 2, 2, 0,
        0, 3, 3, 3, 0, 3, 3, 3, 0,
        0, 4, 4, 4, 0, 4, 4, 4, 0
    };

    configure(ngenes, ngroups);
    auto threads = GetParam();

    scran::differential_analysis::compute_min_rank(ngenes, ngroups, effects.data(), ptrs, threads);
    for (size_t i = 0; i < ngroups; ++i) {
        compare_vectors(std::vector<double>{4, 3, 2, 1}, ngenes, output.data() + i * ngenes); // reversed, for maximum effect sizes.
    }
}

TEST_P(ComputeMinRankTest, LessBasic) {
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
    auto threads = GetParam();

    scran::differential_analysis::compute_min_rank(ngenes, ngroups, effects.data(), ptrs, threads);
    compare_vectors(std::vector<double>{1, 2, 1, 3}, ngenes, output.data());
    compare_vectors(std::vector<double>{2, 3, 1, 1}, ngenes, output.data() + ngenes);
    compare_vectors(std::vector<double>{1, 1, 2, 4}, ngenes, output.data() + ngenes * 2);
}

TEST_P(ComputeMinRankTest, Missing) {
    size_t ngenes = 4, ngroups = 3;
    auto n = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> effects { 
        0, 1, 2, 3, 0, 3, 1, n, 0,
        0, n, 4, 2, 0, 2, 2, n, 0,
        0, 3, 3, n, 0, 4, 3, n, 0,
        0, 4, 1, 1, 0, n, 4, n, 0
     /* 1  1  1  2  2  2  3  3  3 => comparisons for each group */
    };
    for (auto& e : effects) { e *= -1; }

    /* Implicitly, the ranks become:
        0, 1, 2, 3, 0, 2, 1, n, 0,
        0, X, 4, 2, 0, 1, 2, n, 0,
        0, 2, 3, n, 0, 3, 3, n, 0,
        0, 3, 1, 1, 0, X, 4, n, 0
     * after we remove the NA and promote all subsequent entries.
     */

    configure(ngenes, ngroups);
    auto threads = GetParam();

    scran::differential_analysis::compute_min_rank(ngenes, ngroups, effects.data(), ptrs, threads);
    compare_vectors(std::vector<double>{1, 4, 2, 1}, ngenes, output.data());
    compare_vectors(std::vector<double>{2, 1, 3, 1}, ngenes, output.data() + ngenes);
    compare_vectors(std::vector<double>{1, 2, 3, 4}, ngenes, output.data() + ngenes * 2);
}

INSTANTIATE_TEST_SUITE_P(
    ComputeMinRank,
    ComputeMinRankTest,
    ::testing::Values(1, 3) // number of threads
);

// Also checking that our multiple min_rank variants are consistent
// with each other, especially when threading gets involved.
class ComputeMinRankTestThreaded : public ::testing::TestWithParam<std::tuple<int, int, bool> > {
protected:
    std::vector<double*> configure(int ngenes, int ngroups, double* ptr) {
        std::vector<double*> ptrs;
        for (int g = 0; g < ngroups; ++g, ptr += ngenes) {
            ptrs.push_back(ptr);
        }
        return ptrs;
    }
};

TEST_P(ComputeMinRankTestThreaded, Consistency) {
    auto param = GetParam();
    size_t ngenes = 20;
    size_t ngroups = std::get<0>(param);
    size_t nthreads = std::get<1>(param);
    bool add_nans = std::get<2>(param);

    std::mt19937_64 rng(/* seed */ ngroups * nthreads + add_nans);
    std::vector<double> effects(ngenes * ngroups * ngroups);
    std::normal_distribution<double> dist;
    std::uniform_real_distribution<double> unif;
    for (auto& e : effects) {
        e = dist(rng);
        if (add_nans && unif(rng) <= 0.05) {
            e = std::numeric_limits<double>::quiet_NaN();
        }
    }

    std::vector<double> ref_output(ngroups * ngenes);
    std::vector<double*> ref_ptrs = configure(ngenes, ngroups, ref_output.data());
    scran::differential_analysis::compute_min_rank(ngenes, ngroups, effects.data(), ref_ptrs, 1);

    std::vector<double> threaded_output(ngroups * ngenes);
    std::vector<double*> threaded_ptrs = configure(ngenes, ngroups, threaded_output.data());
    scran::differential_analysis::compute_min_rank(ngenes, ngroups, effects.data(), threaded_ptrs, nthreads);
    EXPECT_EQ(threaded_output, ref_output);

    std::vector<double> pergroup_output(ngroups * ngenes);
    for (size_t g = 0; g < ngroups; ++g) {
        std::vector<double> copy(ngroups * ngenes);
        for (size_t i = 0; i < ngenes; ++i) {
            auto base = effects.data() + i * ngroups * ngroups + g * ngroups;
            std::copy(base, base + ngroups, copy.data() + i * ngroups);
        }
        scran::differential_analysis::compute_min_rank(ngenes, ngroups, g, copy.data(), pergroup_output.data() + g * ngenes, nthreads);
    }
    EXPECT_EQ(pergroup_output, ref_output);
}

INSTANTIATE_TEST_SUITE_P(
    ComputeMinRankThreaded,
    ComputeMinRankTestThreaded,
    ::testing::Combine(
        ::testing::Values(1, 2, 3), // number of threads
        ::testing::Values(2, 3, 4, 5), // number of groups
        ::testing::Values(false, true) // whether to spike in NaN's.
    )
);

/*********************************************/

TEST(SummarizeEffects, Basic) {
    size_t ngenes = 11, ngroups = 3;
    std::vector<double> stuff(ngroups * ngroups * ngenes);
    std::mt19937_64 rng(12345);
    std::normal_distribution ndist;
    for (auto& s : stuff) {
        s = ndist(rng);
    }

    scran::SummarizeEffects summarizer;
    auto res = summarizer.run(ngenes, ngroups, stuff.data());
    EXPECT_EQ(res.size(), scran::differential_analysis::n_summaries);

    for (const auto& r : res) {
        EXPECT_EQ(r.size(), ngroups);

        // Different summaries for different groups.
        EXPECT_NE(r[0], r[1]);
        EXPECT_NE(r[0], r[2]);
    }

    // Summaries are actually filled.
    for (size_t g = 0; g < ngroups; ++g) {
        for (int r = 0; r < scran::differential_analysis::n_summaries; ++r) {
            const auto& x = res[r][g];
            EXPECT_EQ(x.size(), ngenes);
            bool nonzero = false;
            for (auto v : x) {
                if (v != 0) {
                    nonzero = true;
                }
            }
            EXPECT_TRUE(nonzero);
        }

        // Min, max and mean make sense.
        const auto& min_vec = res[scran::differential_analysis::MIN][g];
        const auto& max_vec = res[scran::differential_analysis::MAX][g];
        const auto& mean_vec = res[scran::differential_analysis::MEAN][g];
        const auto& med_vec = res[scran::differential_analysis::MEDIAN][g];
        for (size_t i = 0; i < ngenes; ++i) {
            EXPECT_TRUE(min_vec[i] <= mean_vec[i]);
            EXPECT_TRUE(max_vec[i] >= mean_vec[i]);
            EXPECT_TRUE(min_vec[i] <= med_vec[i]);
            EXPECT_TRUE(max_vec[i] >= med_vec[i]);
        }

        // Minimum rank makes sense.
        const auto& ranks = res[scran::differential_analysis::MIN_RANK][g];
        for (auto x : ranks) {
            EXPECT_TRUE(x >= 1);
            EXPECT_TRUE(x <= ngenes);
        }
    }

    // Same results with multiple threads.
    summarizer.set_num_threads(3);
    auto res2 = summarizer.run(ngenes, ngroups, stuff.data());
    EXPECT_EQ(res, res2);
}

TEST(SummarizeEffects, None) {
    size_t ngenes = 0, ngroups = 3;
    std::vector<double> stuff;

    scran::SummarizeEffects summarizer;
    summarizer.set_compute_min(false).set_compute_max(false).set_compute_median(false).set_compute_min_rank(false).set_compute_mean(false);
    auto res = summarizer.run(ngenes, ngroups, stuff.data());

    EXPECT_EQ(res.size(), scran::differential_analysis::n_summaries);
    for (const auto& r : res) {
        EXPECT_EQ(r.size(), 0);
    }
}
