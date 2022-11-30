#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "tatami/base/Matrix.hpp"
#include "tatami/base/DelayedBind.hpp"
#include "tatami/base/DelayedSubset.hpp"
#include "tatami/utils/convert_to_dense.hpp"
#include "tatami/utils/convert_to_sparse.hpp"
#include "scran/differential_analysis/PairwiseEffects.hpp"

#include "../utils/compare_almost_equal.h"
#include "utils.h"

class PairwiseEffectsTestCore : public DifferentialAnalysisTestCore {
protected:
    struct ReferenceResult {
        std::vector<std::vector<double> > means, detected;
        std::vector<double> paired_cohen;
        std::vector<double> paired_auc;
        std::vector<double> paired_lfc;
        std::vector<double> paired_delta_detected;
    };

    static ReferenceResult simple_reference(const tatami::NumericMatrix* mat, const int* group, int ngroups, double threshold) {
        size_t ngenes = mat->nrow();

        scran::differential_analysis::EffectsCalculator runner(1, threshold);
        EffectsOverlord ova(true, ngenes, ngroups);
        auto state = runner.run(mat, group, ngroups, ova);

        ReferenceResult output;
        output.means.resize(ngroups, std::vector<double>(ngenes));
        output.detected = output.means;
        output.paired_cohen.resize(ngroups * ngroups * ngenes);
        output.paired_auc = std::move(ova.store);
        output.paired_lfc = output.paired_cohen;
        output.paired_delta_detected = output.paired_cohen;

        for (size_t g = 0; g < ngenes; ++g) {
            size_t in_offset = g * ngroups;
            size_t out_offset = g * ngroups * ngroups;
            scran::differential_analysis::compute_pairwise_cohens_d(state.means.data() + in_offset, state.variances.data() + in_offset, state.level_size, ngroups, 1, threshold, output.paired_cohen.data() + out_offset);
            scran::differential_analysis::compute_pairwise_simple_diff(state.means.data() + in_offset, state.level_size, ngroups, 1, output.paired_lfc.data() + out_offset);
            scran::differential_analysis::compute_pairwise_simple_diff(state.detected.data() + in_offset, state.level_size, ngroups, 1, output.paired_delta_detected.data() + out_offset);

            for (int i = 0; i < ngroups; ++i) {
                output.means[i][g] = state.means[in_offset + i];
                output.detected[i][g] = state.detected[in_offset + i];
            }
        }

        return output;
    }

    template<class Result>
    static void compare_basic(const Result& res, const Result& other) {
        size_t ngroups = res.means.size();
        EXPECT_EQ(other.means.size(), ngroups);
        EXPECT_EQ(other.detected.size(), ngroups);
        EXPECT_EQ(res.detected.size(), ngroups);

        for (int l = 0; l < ngroups; ++l) {
            EXPECT_EQ(other.means[l].size(), 1);
            EXPECT_EQ(other.detected[l].size(), 1);
            compare_almost_equal(res.means[l][0], other.means[l][0]);
            compare_almost_equal(res.detected[l][0], other.detected[l][0]);
        }
    }

    template<class Result>
    static void compare_blocked(const Result& res, const Result& other) {
        size_t ngroups = res.means.size();
        EXPECT_EQ(other.means.size(), ngroups);
        EXPECT_EQ(other.detected.size(), ngroups);
        EXPECT_EQ(res.detected.size(), ngroups);

        for (int l = 0; l < ngroups; ++l) {
            size_t nblocks = res.means[l].size();
            EXPECT_EQ(res.detected[l].size(), nblocks);
            EXPECT_EQ(other.means[l].size(), nblocks);
            EXPECT_EQ(other.detected[l].size(), nblocks);

            for (int b = 0; b < nblocks; ++b) {
                compare_almost_equal(res.means[l][b], other.means[l][b]);
                compare_almost_equal(res.detected[l][b], other.detected[l][b]);
            }
        }
    }

    template<class Result>
    static void compare_effects(const Result& res, const Result& other, bool do_auc) {
        compare_almost_equal(res.cohen, other.cohen);
        compare_almost_equal(res.lfc, other.lfc);
        compare_almost_equal(res.delta_detected, other.delta_detected);
        if (do_auc) {
            compare_almost_equal(res.auc, other.auc);
        }
    }

    template<class Effects>
    static void check_effects(size_t ngenes, size_t group, const Effects& effects, bool do_auc = false, bool has_boundaries = false, double lower = 0, double upper = 0) {
        EXPECT_EQ(effects.size(), group * group * ngenes);

        for (size_t g = 0; g < ngenes; ++g) {
            auto start = effects.data() + g * group * group;

            // Diagonal effects are still zero.
            for (size_t n = 0; n < group; ++n) {
                EXPECT_EQ(start[n * group + n], 0);
            }

            // Check that the effects are symmetric.
            for (size_t n = 0; n < group; ++n) {
                for (size_t o = 0; o < n; ++o) {
                    if (do_auc) {
                        EXPECT_FLOAT_EQ(start[n * group + o], 1 - start[o * group + n]);
                    } else {
                        EXPECT_FLOAT_EQ(start[n * group + o], -start[o * group + n]);
                    }
                }
            }

            if (has_boundaries) {
                for (size_t n = 0; n < group; ++n) {
                    for (size_t o = 0; o < group; ++o) {
                        auto current = start[n * group + o];
                        EXPECT_TRUE(current >= lower);
                        EXPECT_TRUE(current <= upper);
                    }
                }
            }
        }
    }

    template<class Effects>
    void at_least_one_nonzero(size_t ngenes, size_t group, const Effects& effects, bool global = false) {
        if (!global) {
            for (size_t g = 0; g < ngenes; ++g) {
                auto start = effects.data() + g * group * group;
                bool at_least_one_nonzero = false;
                for (size_t i = 0; i < group * group; ++i) {
                    if (start[i] != 0) {
                        at_least_one_nonzero = true;
                        break;
                    }
                }
                EXPECT_TRUE(at_least_one_nonzero);
            }
        } else {
            bool at_least_one_nonzero = false;
            for (size_t g = 0; g < ngenes; ++g) {
                auto start = effects.data() + g * group * group;
                for (size_t i = 0; i < group * group; ++i) {
                    if (start[i] != 0) {
                        return;
                    }
                }
            }
            EXPECT_TRUE(at_least_one_nonzero);
        }
    }
};

/*********************************************/

class PairwiseEffectsUnblockedTest : public ::testing::TestWithParam<std::tuple<int, double, int> >, public PairwiseEffectsTestCore {
protected:
    void SetUp() {
        assemble();
    }
};

TEST_P(PairwiseEffectsUnblockedTest, Reference) {
    auto param = GetParam();
    auto ngroups = std::get<0>(param);
    std::vector<int> groupings = create_groupings(dense_row->ncol(), ngroups);

    scran::PairwiseEffects chd;
    auto threshold = std::get<1>(param);
    chd.set_threshold(threshold);
    auto nthreads = std::get<2>(param);
    chd.set_num_threads(nthreads);

    auto res = chd.run(dense_row.get(), groupings.data());
    auto ref = simple_reference(dense_row.get(), groupings.data(), ngroups, threshold);

    // Comparing the statistics.
    EXPECT_EQ(res.means.size(), ngroups);
    EXPECT_EQ(res.detected.size(), ngroups);
    for (int g = 0; g < ngroups; ++g) {
        EXPECT_EQ(res.means[g].size(), 1);
        EXPECT_EQ(res.detected[g].size(), 1);
        EXPECT_EQ(res.means[g][0], ref.means[g]);
        EXPECT_EQ(res.detected[g][0], ref.detected[g]);
    }

    EXPECT_EQ(res.cohen, ref.paired_cohen);
    EXPECT_EQ(res.lfc, ref.paired_lfc);
    EXPECT_EQ(res.delta_detected, ref.paired_delta_detected);
    EXPECT_EQ(res.auc, ref.paired_auc);
}

INSTANTIATE_TEST_CASE_P(
    PairwiseEffectsUnblocked,
    PairwiseEffectsUnblockedTest,
    ::testing::Combine(
        ::testing::Values(2, 3, 4, 5), // number of clusters
        ::testing::Values(0, 0.5), // threshold to use
        ::testing::Values(1, 3) // number of threads
    )
);

/*********************************************/

class PairwiseEffectsBlockedTest : public ::testing::TestWithParam<std::tuple<int, int, int> >, public PairwiseEffectsTestCore {
protected:
    static constexpr size_t nrows = 100;
    static constexpr size_t ncols = 50;

    void SetUp() {
        assemble(nrows, ncols);
    }
};

TEST_P(PairwiseEffectsBlockedTest, Blocked) {
    auto param = GetParam();
    auto ngroups = std::get<0>(param);
    auto nblocks = std::get<1>(param);
    auto nthreads = std::get<2>(param);

    scran::PairwiseEffects chd;
    chd.set_num_threads(nthreads);

    auto groups = create_groupings(ncols, ngroups);
    auto blocks = create_blocks(ncols, nblocks);
    auto full = chd.run_blocked(sparse_row.get(), groups.data(), blocks.data());

    std::vector<double> ref_cohen(ngroups * ngroups * nrows);
    auto ref_auc = ref_cohen;
    auto ref_lfc = ref_cohen;
    auto ref_delta_detected = ref_cohen;
    std::vector<int> total_weights(ngroups * ngroups);

    std::vector<std::vector<std::vector<double> > > ref_means(ngroups, 
        std::vector<std::vector<double> >(nblocks, std::vector<double>(nrows)));
    auto ref_detected = ref_means;

    for (int b = 0; b < nblocks; ++b) {
        std::vector<int> subset;
        std::vector<int> subgroups;
        std::vector<int> subcount(ngroups);
        for (int i = 0; i < ncols; ++i) {
            if (blocks[i] == b) {
                subset.push_back(i);
                subgroups.push_back(groups[i]);
                ++subcount[groups[i]];
            }
        }

        auto sub = tatami::make_DelayedSubset<1>(dense_row, std::move(subset));
        auto res = chd.run(sub.get(), subgroups.data());

        for (size_t i = 0; i < nrows; ++i) {
            for (int g1 = 0; g1 < ngroups; ++g1) {
                for (int g2 = 0; g2 < ngroups; ++g2) {
                    size_t offset = i * ngroups * ngroups + g1 * ngroups + g2;
                    double weight = subcount[g1] * subcount[g2];
                    ref_cohen[offset] += weight * res.cohen[offset];
                    ref_lfc[offset] += weight * res.lfc[offset];
                    ref_delta_detected[offset] += weight * res.delta_detected[offset];
                    ref_auc[offset] += weight * res.auc[offset];
                }
            }

            for (int g = 0; g < ngroups; ++g) {
                ref_means[g][b][i] = res.means[g][0][i];
                ref_detected[g][b][i] = res.detected[g][0][i];
            }
        }

        for (int g1 = 0; g1 < ngroups; ++g1) {
            for (int g2 = 0; g2 < ngroups; ++g2) {
                total_weights[g1 * ngroups + g2] += subcount[g1] * subcount[g2];
            }
        }
    }

    for (size_t i = 0; i < nrows; ++i) {
        auto offset = i * ngroups * ngroups;
        for (int g = 0; g < ngroups * ngroups; ++g) {
            ref_cohen[offset + g] /= total_weights[g];
            ref_lfc[offset + g] /= total_weights[g];
            ref_auc[offset + g] /= total_weights[g];
            ref_delta_detected[offset + g] /= total_weights[g];
        }
    }

    // Alright, running all the tests.
    compare_almost_equal(ref_cohen, full.cohen);
    compare_almost_equal(ref_lfc, full.lfc);
    compare_almost_equal(ref_delta_detected, full.delta_detected);
    compare_almost_equal(ref_auc, full.auc);

    for (int g = 0; g < ngroups; ++g) {
        for (int b = 0; b < nblocks; ++b) {
            EXPECT_EQ(ref_means[g][b], full.means[g][b]);
            EXPECT_EQ(ref_detected[g][b], full.detected[g][b]);
        }
    }
}

INSTANTIATE_TEST_CASE_P(
    PairwiseEffectsBlocked,
    PairwiseEffectsBlockedTest,
    ::testing::Combine(
        ::testing::Values(2, 3, 4, 5), // number of clusters
        ::testing::Values(1, 2, 3), // number of blocks
        ::testing::Values(1, 3) // number of threads
    )
);

/*********************************************/

class PairwiseEffectsScenarioTest : public ::testing::TestWithParam<std::tuple<int, bool> >, public PairwiseEffectsTestCore {
    void SetUp() {
        assemble();
    }
};

TEST_P(PairwiseEffectsScenarioTest, Self) {
    // Replicating the same matrix 'ngroups' times.
    int ngroups = std::get<0>(GetParam());
    std::vector<std::shared_ptr<tatami::NumericMatrix> > stuff;
    for (int i = 0; i < ngroups; ++i) {
        stuff.push_back(dense_row);
    }
    auto combined = tatami::make_DelayedBind<1>(std::move(stuff));

    // Creating two groups; second group can be larger than the first, to check
    // for correct behavior w.r.t. imbalanced groups.
    size_t NC = dense_row->ncol();
    std::vector<int> groupings(NC * ngroups);
    std::fill(groupings.begin(), groupings.begin() + NC, 0);
    std::fill(groupings.begin() + NC, groupings.end(), 1); 

    scran::PairwiseEffects chd;
    bool do_auc = std::get<1>(GetParam());
    chd.set_compute_auc(do_auc); // false, if we want to check the running implementations.
    auto res = chd.run(combined.get(), groupings.data());

    // All AUCs should be 0.5, all Cohen/LFC/delta-d's should be 0.
    size_t ngenes = dense_row->nrow();
    for (size_t g = 0; g < ngenes; ++g) {
        for (int l = 0; l < 2; ++l) {
            for (int l2 = 0; l2 < 2; ++l2) {
                if (l == l2) {
                    break;
                }

                // Only 2 groups here, despite the 'ngroups' variable!
                size_t offset = g * 2 * 2 + l * 2 + l2;  

                // Handle some numerical imprecision...
                EXPECT_TRUE(std::abs(res.cohen[offset]) < 1e-10);
                EXPECT_TRUE(std::abs(res.lfc[offset]) < 1e-10);
                EXPECT_TRUE(std::abs(res.delta_detected[offset]) < 1e-10);
                if (do_auc) {
                    EXPECT_FLOAT_EQ(res.auc[offset], 0.5);
                }
            }
        }
    }
}

TEST_P(PairwiseEffectsScenarioTest, Thresholds) {
    auto ngroups = std::get<0>(GetParam());
    std::vector<int> groupings = create_groupings(dense_row->ncol(), ngroups);

    scran::PairwiseEffects chd;
    bool do_auc = std::get<1>(GetParam());
    chd.set_compute_auc(do_auc); // false, if we want to check the running implementations.
    auto ref = chd.run(dense_row.get(), groupings.data());
    auto out = chd.set_threshold(1).run(dense_row.get(), groupings.data());

    bool some_diff = false;
    size_t ngenes = dense_row->nrow();
    for (size_t g = 0; g < ngenes; ++g) {
        for (int l = 0; l < ngroups; ++l) {
            for (int l2 = 0; l2 < ngroups; ++l2) {
                if (l == l2) {
                    break;
                }

                size_t offset = g * ngroups * ngroups + l * ngroups + l2;  
                EXPECT_TRUE(ref.cohen[offset] > out.cohen[offset]);

                if (do_auc) {
                    some_diff |= (ref.auc[offset] > out.auc[offset]);

                    // '>' is not guaranteed due to imprecision with ranks... but (see below).
                    EXPECT_TRUE(ref.auc[offset] >= out.auc[offset]); 
                }
            }
        }
    }

    if (do_auc) {
        EXPECT_TRUE(some_diff); // (from above)... at least one is '>', hopefully.
    }
}

TEST_P(PairwiseEffectsScenarioTest, Missing) {
    auto ngroups = std::get<0>(GetParam());
    std::vector<int> groupings = create_groupings(dense_row->ncol(), ngroups);

    scran::PairwiseEffects chd;
    bool do_auc = std::get<1>(GetParam());
    chd.set_compute_auc(do_auc); // false, if we want to check the running implementations.
    auto ref = chd.run(dense_row.get(), groupings.data());
    
    for (auto& g : groupings) { // 0 is the missing group.
        ++g;
    }
    auto lost = chd.run(dense_row.get(), groupings.data());

    // Everything should be NaN.
    size_t ngenes = dense_row->nrow();
    int ngroups_p1 = ngroups + 1;

    for (size_t g = 0; g < ngenes; ++g) {
        for (int l2 = 1; l2 < ngroups_p1; ++l2) {
            // For the comparisons from group 0 to the others.
            size_t offset = g * ngroups_p1 * ngroups_p1 + l2;  
            EXPECT_TRUE(std::isnan(lost.cohen[offset]));
            EXPECT_TRUE(std::isnan(lost.lfc[offset]));
            EXPECT_TRUE(std::isnan(lost.delta_detected[offset]));
            if (do_auc) {
                EXPECT_TRUE(std::isnan(lost.auc[offset]));
            }

            // For the comparisons in the other direction.
            offset = g * ngroups_p1 * ngroups_p1 + l2 * ngroups_p1;  
            EXPECT_TRUE(std::isnan(lost.cohen[offset]));
            EXPECT_TRUE(std::isnan(lost.lfc[offset]));
            EXPECT_TRUE(std::isnan(lost.delta_detected[offset]));
            if (do_auc) {
                EXPECT_TRUE(std::isnan(lost.auc[offset]));
            }
        }
    }

    // Other metrics should be the same as usual.
    for (size_t g = 0; g < ngenes; ++g) {
        for (int l = 0; l < ngroups; ++l) {
            size_t ref_offset = g * ngroups * ngroups + l * ngroups;  
            size_t lost_offset = g * ngroups_p1 * ngroups_p1 + (l + 1) * ngroups_p1 + 1; // skip group 0, and also the NaN in the comparison against group 0.

            EXPECT_EQ(
                std::vector<double>(ref.cohen.begin() + ref_offset, ref.cohen.begin() + ref_offset + ngroups), 
                std::vector<double>(lost.cohen.begin() + lost_offset, lost.cohen.begin() + lost_offset + ngroups)
            );

            EXPECT_EQ(
                std::vector<double>(ref.lfc.begin() + ref_offset, ref.lfc.begin() + ref_offset + ngroups), 
                std::vector<double>(lost.lfc.begin() + lost_offset, lost.lfc.begin() + lost_offset + ngroups)
            );

            EXPECT_EQ(
                std::vector<double>(ref.delta_detected.begin() + ref_offset, ref.delta_detected.begin() + ref_offset + ngroups), 
                std::vector<double>(lost.delta_detected.begin() + lost_offset, lost.delta_detected.begin() + lost_offset + ngroups)
            );

            EXPECT_EQ(
                std::vector<double>(ref.cohen.begin() + ref_offset, ref.cohen.begin() + ref_offset + ngroups), 
                std::vector<double>(lost.cohen.begin() + lost_offset, lost.cohen.begin() + lost_offset + ngroups)
            );
        }
    }
}

TEST_P(PairwiseEffectsScenarioTest, BlockConfounded) {
    auto NC = dense_row->ncol();
    auto ngroups = std::get<0>(GetParam());
    std::vector<int> groupings = create_groupings(NC, ngroups);

    // Block is fully confounded with one group.
    std::vector<int> blocks(NC);
    for (size_t c = 0; c < NC; ++c) {
        blocks[c] = groupings[c] == 0;
    }

    scran::PairwiseEffects chd;
    bool do_auc = std::get<1>(GetParam());
    chd.set_compute_auc(do_auc); // false, if we want to check the running implementations.
    auto comres = chd.run_blocked(dense_row.get(), groupings.data(), blocks.data());

    // First group should only be NaN's.
    size_t ngenes = dense_row->nrow();
    for (size_t g = 0; g < ngenes; ++g) {
        for (int l2 = 1; l2 < ngroups; ++l2) {
            // For the comparisons from group 0 to the others.
            size_t offset = g * ngroups * ngroups + l2;  
            EXPECT_TRUE(std::isnan(comres.cohen[offset]));
            EXPECT_TRUE(std::isnan(comres.lfc[offset]));
            EXPECT_TRUE(std::isnan(comres.delta_detected[offset]));
            if (do_auc) {
                EXPECT_TRUE(std::isnan(comres.auc[offset]));
            }

            // For the comparisons in the other direction.
            offset = g * ngroups * ngroups + l2 * ngroups;  
            EXPECT_TRUE(std::isnan(comres.cohen[offset]));
            EXPECT_TRUE(std::isnan(comres.lfc[offset]));
            EXPECT_TRUE(std::isnan(comres.delta_detected[offset]));
            if (do_auc) {
                EXPECT_TRUE(std::isnan(comres.auc[offset]));
            }
        }
    }

    if (ngroups > 2) {
        // Excluding the group and running on the remaining samples.
        std::vector<int> subgroups;
        std::vector<int> keep;
        for (size_t c = 0; c < NC; ++c) {
            auto g = groupings[c];
            if (g != 0) {
                subgroups.push_back(g - 1);
                keep.push_back(c);
            }
        }

        auto sub = tatami::make_DelayedSubset<1>(dense_row, std::move(keep));
        auto ref = chd.run(sub.get(), subgroups.data()); 
        int ngroups_m1 = ngroups - 1;

        for (size_t g = 0; g < ngenes; ++g) {
            for (int l = 0; l < ngroups_m1; ++l) {
                size_t ref_offset = g * ngroups_m1 * ngroups_m1 + l * ngroups_m1;  
                size_t comres_offset = g * ngroups * ngroups + (l + 1) * ngroups + 1; // skip group 0 as well as the NaN in the comparison against group 0.

                EXPECT_EQ(
                    std::vector<double>(ref.cohen.begin() + ref_offset, ref.cohen.begin() + ref_offset + ngroups_m1), 
                    std::vector<double>(comres.cohen.begin() + comres_offset, comres.cohen.begin() + comres_offset + ngroups_m1)
                );

                EXPECT_EQ(
                    std::vector<double>(ref.lfc.begin() + ref_offset, ref.lfc.begin() + ref_offset + ngroups_m1), 
                    std::vector<double>(comres.lfc.begin() + comres_offset, comres.lfc.begin() + comres_offset + ngroups_m1)
                );

                EXPECT_EQ(
                    std::vector<double>(ref.delta_detected.begin() + ref_offset, ref.delta_detected.begin() + ref_offset + ngroups_m1), 
                    std::vector<double>(comres.delta_detected.begin() + comres_offset, comres.delta_detected.begin() + comres_offset + ngroups_m1)
                );

                EXPECT_EQ(
                    std::vector<double>(ref.cohen.begin() + ref_offset, ref.cohen.begin() + ref_offset + ngroups_m1), 
                    std::vector<double>(comres.cohen.begin() + comres_offset, comres.cohen.begin() + comres_offset + ngroups_m1)
                );
            }
        }
    }
}

INSTANTIATE_TEST_CASE_P(
    PairwiseEffectsScenario,
    PairwiseEffectsScenarioTest,
    ::testing::Combine(
        ::testing::Values(2, 3, 4, 5), // number of clusters
        ::testing::Values(false, true) // with or without the AUC?
    )
);
