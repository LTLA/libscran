#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "tatami/base/Matrix.hpp"
#include "tatami/base/DelayedBind.hpp"
#include "tatami/base/DelayedSubset.hpp"
#include "tatami/utils/convert_to_dense.hpp"
#include "tatami/utils/convert_to_sparse.hpp"
#include "scran/differential_analysis/PairwiseEffects.hpp"

#include "../data/data.h"
#include "../utils/compare_almost_equal.h"

class PairwiseEffectsTestCore {
protected:
    std::shared_ptr<tatami::NumericMatrix> dense_row, dense_column, sparse_row, sparse_column;

    void assemble() {
        dense_row = std::unique_ptr<tatami::NumericMatrix>(new tatami::DenseRowMatrix<double>(sparse_nrow, sparse_ncol, sparse_matrix));
        dense_column = tatami::convert_to_dense(dense_row.get(), 1);
        sparse_row = tatami::convert_to_sparse(dense_row.get(), 0);
        sparse_column = tatami::convert_to_sparse(dense_row.get(), 1);
    }

    static std::vector<int> create_groupings(size_t n, int ngroups) {
        std::vector<int> groupings(n);
        for (size_t g = 0; g < groupings.size(); ++g) {
            groupings[g] = g % ngroups;
        }
        return groupings;
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
        bool at_least_one_nonzero = false;

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

            if (!at_least_one_nonzero) {
                for (size_t i = 0; i < group * group; ++i) {
                    if (start[i] != 0) {
                        at_least_one_nonzero = true;
                        break;
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

        EXPECT_TRUE(at_least_one_nonzero);
    }

};

/*********************************************/

class PairwiseEffectsTest : public ::testing::TestWithParam<std::tuple<int, bool, int> >, public PairwiseEffectsTestCore {
protected:
    void SetUp() {
        assemble();
    }
};

TEST_P(PairwiseEffectsTest, Basics) {
    auto ngroups = std::get<0>(GetParam());
    std::vector<int> groupings = create_groupings(dense_row->ncol(), ngroups);

    scran::PairwiseEffects chd;
    bool do_auc = std::get<1>(GetParam());
    chd.set_compute_auc(do_auc); // false, if we want to check the running implementations.
    auto res = chd.run(dense_row.get(), groupings.data());

    auto nthreads = std::get<2>(GetParam());
    chd.set_num_threads(nthreads);

    if (nthreads == 1) {
        size_t ngenes = dense_row->nrow();

        // Running cursory checks on the metrics.
        EXPECT_EQ(res.means.size(), ngroups);
        EXPECT_EQ(res.detected.size(), ngroups);

        for (size_t g = 0; g < ngenes; ++g) {
            std::vector<double> buf(dense_row->ncol());
            auto ptr = dense_row->row(g, buf.data());

            for (int l = 0; l < ngroups; ++l) {
                double curmean = 0, curdetected = 0;
                int count = 0;
                for (int i = l; i < dense_row->ncol(); i += ngroups) {
                    curmean += ptr[i];
                    curdetected += (ptr[i] > 0);
                    ++count;
                }

                curmean /= count;
                EXPECT_EQ(curmean, res.means[l][0][g]);
                curdetected /= count;
                EXPECT_EQ(curdetected, res.detected[l][0][g]);
            }
        }

        // Running some further checks on the effects.
        check_effects(ngenes, ngroups, res.cohen);
        check_effects(ngenes, ngroups, res.lfc);
        check_effects(ngenes, ngroups, res.delta_detected, false, true, -1, 1);
        if (do_auc) {
            check_effects(ngenes, ngroups, res.auc, true, true, 0, 1);
        }
    } else {
        // Comparing to the same call, but parallelized.
        auto res1 = chd.run(dense_row.get(), groupings.data());
        compare_basic(res, res1);
        compare_effects(res, res1, do_auc);
    }

    // Comparing to other implementations. 
    auto res2 = chd.run(sparse_row.get(), groupings.data());
    compare_basic(res, res2);
    compare_effects(res, res2, do_auc);

    auto res3 = chd.run(dense_column.get(), groupings.data());
    compare_basic(res, res3);
    compare_effects(res, res3, do_auc);

    auto res4 = chd.run(sparse_column.get(), groupings.data());
    compare_basic(res, res4);
    compare_effects(res, res4, do_auc);
}

INSTANTIATE_TEST_CASE_P(
    PairwiseEffects,
    PairwiseEffectsTest,
    ::testing::Combine(
        ::testing::Values(2, 3, 4, 5), // number of clusters
        ::testing::Values(false, true), // with or without the AUC?
        ::testing::Values(1, 3) // number of threads
    )
);

/*********************************************/

class PairwiseEffectsBlockTest : public ::testing::TestWithParam<std::tuple<int, bool, bool, int> >, public PairwiseEffectsTestCore {
protected:
    void SetUp() {
        assemble();
    }

    static bool is_sandwiched(double bottombread, double meat, double topbread) {
        if (bottombread < topbread) {
            if (bottombread <= meat && meat <= topbread) {
                return true;
            }
        } else {
            if (bottombread >= meat && meat >= topbread) {
                return true;
            }
        }
        return false;
    }
};

TEST_P(PairwiseEffectsBlockTest, Blocked) {
    auto param = GetParam();
    auto ngroups = std::get<0>(param);
    bool do_auc = std::get<1>(param);
    bool do_zero = std::get<2>(param);
    auto nthreads = std::get<3>(param);

    std::shared_ptr<tatami::NumericMatrix> other_dense_row, other_dense_column, other_sparse_row, other_sparse_column;
    std::vector<int> groupings;
    auto NR = dense_row->nrow();
    auto NC = dense_row->ncol();

    if (!do_zero) {
        // Check that everything is more or less computed correctly,
        // by duplicating the matrices and blocking on them.
        other_dense_row = dense_row;
        other_dense_column = dense_column;
        other_sparse_row = sparse_row;
        other_sparse_column = sparse_column;

        // Groupings may or may not be perfectly crossed with block, 
        // depending on whether ngroups is a multiple of NC.
        groupings = create_groupings(NC * 2, ngroups);

    } else {
        // Duplicating grouping exactly for easier testing below.
        auto tmp_groupings = create_groupings(NC, ngroups);
        groupings.insert(groupings.end(), tmp_groupings.begin(), tmp_groupings.end());
        groupings.insert(groupings.end(), tmp_groupings.begin(), tmp_groupings.end());

        // Filling with zeros and checking for suppressed effects.
        std::vector<double> allzeros(NR * NC);
        other_dense_row = std::shared_ptr<tatami::NumericMatrix>(new tatami::DenseRowMatrix<double, int, decltype(allzeros)>(NR, NC, std::move(allzeros)));
        other_dense_column = tatami::convert_to_dense(other_dense_row.get(), 1);
        other_sparse_row = tatami::convert_to_sparse(other_dense_row.get(), 0);
        other_sparse_column = tatami::convert_to_sparse(other_dense_row.get(), 1);
    }

    auto combined_dense_row = tatami::make_DelayedBind<1>(std::vector<std::shared_ptr<tatami::NumericMatrix> >{dense_row, other_dense_row});
    auto combined_dense_column = tatami::make_DelayedBind<1>(std::vector<std::shared_ptr<tatami::NumericMatrix> >{dense_column, other_dense_column});
    auto combined_sparse_row = tatami::make_DelayedBind<1>(std::vector<std::shared_ptr<tatami::NumericMatrix> >{sparse_row, other_sparse_row});
    auto combined_sparse_column = tatami::make_DelayedBind<1>(std::vector<std::shared_ptr<tatami::NumericMatrix> >{sparse_column, other_sparse_column});

    std::vector<int> blocks(groupings.size());
    std::fill(blocks.begin() + NC, blocks.end(), 1);

    scran::PairwiseEffects chd;
    chd.set_compute_auc(do_auc); // false, if we want to check the running implementations.
    auto comres = chd.run_blocked(combined_dense_row.get(), groupings.data(), blocks.data());

    chd.set_num_threads(nthreads);

    if (nthreads == 1) {
        std::vector<int> g1(groupings.begin(), groupings.begin() + NC);
        auto res1 = chd.run(dense_row.get(), g1.data());
        std::vector<int> g2(groupings.begin() + NC, groupings.end());
        auto res2 = chd.run(other_dense_row.get(), g2.data());

        size_t neffects = NR * ngroups * ngroups;
        if (do_zero) {
            // Effects should be shrunk towards zero when we added a block of all-zero elements.
            bool smaller_cohen = true, smaller_lfc = true, smaller_delta = true, smaller_auc = true;
            for (size_t g = 0; g < neffects; ++g) {
                if (!is_sandwiched(0, comres.cohen[g], res1.cohen[g])) {
                    smaller_cohen = false;
                }
                if (!is_sandwiched(0, comres.lfc[g], res1.lfc[g])) {
                    smaller_lfc = false;
                }
                if (!is_sandwiched(0, comres.delta_detected[g], res1.delta_detected[g])) {
                    smaller_delta = false;
                }
                if (do_auc && !is_sandwiched(0.5, comres.auc[g], res1.auc[g])) {
                    smaller_auc = false;
                }
            }

            EXPECT_TRUE(smaller_cohen);
            EXPECT_TRUE(smaller_lfc);
            EXPECT_TRUE(smaller_delta);

            EXPECT_NE(comres.cohen, res1.cohen);
            EXPECT_NE(comres.lfc, res1.lfc);
            EXPECT_NE(comres.delta_detected, res1.delta_detected);

            if (do_auc) {
                EXPECT_TRUE(smaller_auc);
                EXPECT_NE(comres.auc, res1.auc);
            }

        } else {
            // If the number of columns is a multiple of the number of groups,
            // we should get the exact same results. Otherwise we should get
            // results in between the per-block analyses.
            if (NC % ngroups == 0) {
                compare_effects(res1, comres, do_auc);
            } else {
                bool middle_cohen = true, middle_lfc = true, middle_delta = true, middle_auc = true;
                for (size_t g = 0; g < neffects; ++g) {
                    if (!is_sandwiched(res1.cohen[g], comres.cohen[g], res2.cohen[g])) {
                        middle_cohen = false;
                    }
                    if (!is_sandwiched(res1.lfc[g], comres.lfc[g], res2.lfc[g])) {
                        middle_lfc = false;
                    }
                    if (!is_sandwiched(res1.delta_detected[g], comres.delta_detected[g], res2.delta_detected[g])) {
                        middle_delta = false;
                    }
                    if (do_auc && !is_sandwiched(res1.auc[g], comres.auc[g], res2.auc[g])) {
                        middle_auc = false;
                    }
                }

                EXPECT_TRUE(middle_cohen);
                EXPECT_TRUE(middle_lfc);
                EXPECT_TRUE(middle_delta);

                EXPECT_NE(comres.cohen, res1.cohen);
                EXPECT_NE(comres.lfc, res1.lfc);
                EXPECT_NE(comres.delta_detected, res1.delta_detected);

                if (do_auc) {
                    EXPECT_TRUE(middle_auc);
                    EXPECT_NE(comres.auc, res1.auc);
                }
            }
        }

        // Compare metrics and effects
        EXPECT_EQ(comres.means.size(), ngroups);
        EXPECT_EQ(comres.detected.size(), ngroups);

        for (int l = 0; l < ngroups; ++l) {
            compare_almost_equal(comres.means[l][0], res1.means[l][0]);
            compare_almost_equal(comres.detected[l][0], res1.detected[l][0]);
            compare_almost_equal(comres.means[l][1], res2.means[l][0]);
            compare_almost_equal(comres.detected[l][1], res2.detected[l][0]);
        }
    } else {
        // Comparing to the same call, but parallelized.
        auto comres1 = chd.run_blocked(combined_dense_row.get(), groupings.data(), blocks.data());
        compare_blocked(comres, comres1);
        compare_effects(comres, comres1, do_auc);
    }

    auto comres2 = chd.run_blocked(combined_dense_column.get(), groupings.data(), blocks.data());
    compare_blocked(comres, comres2);
    compare_effects(comres, comres2, do_auc);

    auto comres3 = chd.run_blocked(combined_sparse_row.get(), groupings.data(), blocks.data());
    compare_blocked(comres, comres3);
    compare_effects(comres, comres3, do_auc);

    auto comres4 = chd.run_blocked(combined_sparse_column.get(), groupings.data(), blocks.data());
    compare_blocked(comres, comres4);
    compare_effects(comres, comres4, do_auc);
}

INSTANTIATE_TEST_CASE_P(
    PairwiseEffectsBlock,
    PairwiseEffectsBlockTest,
    ::testing::Combine(
        ::testing::Values(2, 3, 4, 5), // number of clusters
        ::testing::Values(false, true), // with or without the AUC?
        ::testing::Values(false, true), // with or without adding zeros?
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

TEST_P(PairwiseEffectsTest, BlockConfounded) {
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
