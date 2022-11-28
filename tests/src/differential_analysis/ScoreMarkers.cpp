#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "tatami/base/Matrix.hpp"
#include "tatami/base/DelayedBind.hpp"
#include "tatami/base/DelayedSubset.hpp"
#include "tatami/utils/convert_to_dense.hpp"
#include "tatami/utils/convert_to_sparse.hpp"
#include "scran/differential_analysis/ScoreMarkers.hpp"
#include "scran/differential_analysis/PairwiseEffects.hpp"
#include "scran/differential_analysis/SummarizeEffects.hpp"

#include "../data/data.h"
#include "../utils/compare_almost_equal.h"

class ScoreMarkersTestCore {
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
};

/*********************************************/

class ScoreMarkersTest : public ::testing::TestWithParam<std::tuple<int, bool, int> >, public ScoreMarkersTestCore {
protected:
    void SetUp() {
        assemble();
    }
    
    template<class Result>
    static void compare_basic(int ngroups, const Result& res, const Result& other) {
        for (int l = 0; l < ngroups; ++l) {
            EXPECT_EQ(other.means[l].size(), 1);
            EXPECT_EQ(other.detected[l].size(), 1);
            compare_almost_equal(res.means[l][0], other.means[l][0]);
            compare_almost_equal(res.detected[l][0], other.detected[l][0]);
        }
    }

    template<class Result>
    static void compare_blocked(int ngroups, int nblocks, const Result& res, const Result& other) {
        for (int l = 0; l < ngroups; ++l) {
            EXPECT_EQ(other.means[l].size(), nblocks);
            EXPECT_EQ(other.detected[l].size(), nblocks);

            for (int b = 0; b < nblocks; ++b) {
                compare_almost_equal(res.means[l][b], other.means[l][b]);
                compare_almost_equal(res.detected[l][b], other.detected[l][b]);
            }
        }
    }

    template<class Result>
    static void compare_effects(int ngroups, const Result& res, const Result& other, bool do_auc) {
        // Don't compare min-rank here, as minor numerical differences
        // can change the ranks by a small amount when effects are tied.
        for (int s = 0; s < scran::differential_analysis::MIN_RANK; ++s) {
            EXPECT_EQ(res.cohen[s].size(), other.cohen[s].size());
            EXPECT_EQ(res.lfc[s].size(), other.lfc[s].size());
            EXPECT_EQ(res.delta_detected[s].size(), other.delta_detected[s].size());

            for (int l = 0; l < ngroups; ++l) {
                compare_almost_equal(res.cohen[s][l], other.cohen[s][l]);
                compare_almost_equal(res.lfc[s][l], other.lfc[s][l]);
                compare_almost_equal(res.delta_detected[s][l], other.delta_detected[s][l]);
            }

            if (do_auc) {
                EXPECT_EQ(res.auc[s].size(), other.auc[s].size());
                for (int l = 0; l < ngroups; ++l) {
                    compare_almost_equal(res.auc[s][l], other.auc[s][l]);
                }
            }
        }
    }

    template<class Effects>
    static void check_effects(size_t ngenes, size_t group, const Effects& effects) {
        for (size_t g = 0; g < ngenes; ++g) {
            double curmin = effects[scran::differential_analysis::MIN][group][g];
            double curmean = effects[scran::differential_analysis::MEAN][group][g];
            double curmed = effects[scran::differential_analysis::MEDIAN][group][g];
            double curmax = effects[scran::differential_analysis::MAX][group][g];
            double currank = effects[scran::differential_analysis::MIN_RANK][group][g];

            EXPECT_TRUE(curmin <= curmean);
            EXPECT_TRUE(curmin <= curmed);
            EXPECT_TRUE(curmax >= curmean);
            EXPECT_TRUE(curmax >= curmed);

            if (curmed > curmin) {
                EXPECT_TRUE(curmean > curmin);
            }
            if (curmed < curmax) {
                EXPECT_TRUE(curmean < curmax);
            }

            EXPECT_TRUE(currank >= 1);
            EXPECT_TRUE(currank <= ngenes);
        }
    }
};

TEST_P(ScoreMarkersTest, Basics) {
    auto ngroups = std::get<0>(GetParam());
    std::vector<int> groupings = create_groupings(dense_row->ncol(), ngroups);

    scran::ScoreMarkers chd;
    bool do_auc = std::get<1>(GetParam());
    chd.set_compute_auc(do_auc); // false, if we want to check the running implementations.
    auto res = chd.run(dense_row.get(), groupings.data());

    auto nthreads = std::get<2>(GetParam());
    chd.set_num_threads(nthreads);

    if (nthreads == 1) {
        size_t ngenes = dense_row->nrow();

        // Running cursory checks on the metrics.
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
        for (int l = 0; l < ngroups; ++l) {
            check_effects(ngenes, l, res.cohen);
            check_effects(ngenes, l, res.lfc);

            check_effects(ngenes, l, res.delta_detected);
            for (size_t g = 0; g < ngenes; ++g) {
                double curmin = res.delta_detected[scran::differential_analysis::MIN][l][g];
                double curmax = res.delta_detected[scran::differential_analysis::MAX][l][g];
                EXPECT_TRUE(curmin >= -1);
                EXPECT_TRUE(curmax <= 1);
            }

            if (do_auc) {
                check_effects(ngenes, l, res.auc);
                for (size_t g = 0; g < ngenes; ++g) {
                    double curmin = res.auc[scran::differential_analysis::MIN][l][g];
                    double curmax = res.auc[scran::differential_analysis::MAX][l][g];
                    EXPECT_TRUE(curmin >= 0);
                    EXPECT_TRUE(curmax <= 1);
                }
            }
        }
    } else {
        // Comparing to the same call, but parallelized.
        auto res1 = chd.run(dense_row.get(), groupings.data());
        compare_basic(ngroups, res, res1);
        compare_effects(ngroups, res, res1, do_auc);
    }

    // Comparing to other implementations. 
    auto res2 = chd.run(sparse_row.get(), groupings.data());
    compare_basic(ngroups, res, res2);
    compare_effects(ngroups, res, res2, do_auc);

    auto res3 = chd.run(dense_column.get(), groupings.data());
    compare_basic(ngroups, res, res3);
    compare_effects(ngroups, res, res3, do_auc);

    auto res4 = chd.run(sparse_column.get(), groupings.data());
    compare_basic(ngroups, res, res4);
    compare_effects(ngroups, res, res4, do_auc);
}

TEST_P(ScoreMarkersTest, Blocked) {
    // Check that everything is more or less computed correctly,
    // by duplicating the matrices and blocking on them.
    auto combined = tatami::make_DelayedBind<1>(std::vector<std::shared_ptr<tatami::NumericMatrix> >{dense_row, dense_row});

    auto NC = dense_row->ncol();
    auto ngroups = std::get<0>(GetParam());
    std::vector<int> groupings = create_groupings(NC * 2, ngroups);
    std::vector<int> blocks(groupings.size());
    std::fill(blocks.begin() + NC, blocks.end(), 1);

    scran::ScoreMarkers chd;
    bool do_auc = std::get<1>(GetParam());
    chd.set_compute_auc(do_auc); // false, if we want to check the running implementations.
    auto comres = chd.run_blocked(combined.get(), groupings.data(), blocks.data());

    auto nthreads = std::get<2>(GetParam());
    chd.set_num_threads(nthreads);

    if (nthreads == 1) {
        std::vector<int> g1(groupings.begin(), groupings.begin() + NC);
        auto res1 = chd.run(dense_row.get(), g1.data());
        std::vector<int> g2(groupings.begin() + NC, groupings.end());
        auto res2 = chd.run(dense_row.get(), g2.data());

        if (NC % ngroups == 0) {
            compare_effects(ngroups, comres, res1, do_auc);
            compare_effects(ngroups, comres, res2, do_auc);
        }

        for (int l = 0; l < ngroups; ++l) {
            compare_almost_equal(comres.means[l][0], res1.means[l][0]);
            compare_almost_equal(comres.detected[l][0], res1.detected[l][0]);
            compare_almost_equal(comres.means[l][1], res2.means[l][0]);
            compare_almost_equal(comres.detected[l][1], res2.detected[l][0]);
        }
    } else {
        // Comparing to the same call, but parallelized.
        auto comres1 = chd.run_blocked(combined.get(), groupings.data(), blocks.data());
        compare_blocked(ngroups, 2, comres, comres1);
        compare_effects(ngroups, comres, comres1, do_auc);
    }

    auto combined2 = tatami::make_DelayedBind<1>(std::vector<std::shared_ptr<tatami::NumericMatrix> >{sparse_row, sparse_row});
    auto comres2 = chd.run_blocked(combined2.get(), groupings.data(), blocks.data());
    compare_blocked(ngroups, 2, comres, comres2);
    compare_effects(ngroups, comres, comres2, do_auc);

    auto combined3 = tatami::make_DelayedBind<1>(std::vector<std::shared_ptr<tatami::NumericMatrix> >{dense_column, dense_column});
    auto comres3 = chd.run_blocked(combined3.get(), groupings.data(), blocks.data());
    compare_blocked(ngroups, 2, comres, comres3);
    compare_effects(ngroups, comres, comres3, do_auc);

    auto combined4 = tatami::make_DelayedBind<1>(std::vector<std::shared_ptr<tatami::NumericMatrix> >{sparse_column, sparse_column});
    auto comres4 = chd.run_blocked(combined4.get(), groupings.data(), blocks.data());
    compare_blocked(ngroups, 2, comres, comres4);
    compare_effects(ngroups, comres, comres4, do_auc);
}

INSTANTIATE_TEST_CASE_P(
    ScoreMarkers,
    ScoreMarkersTest,
    ::testing::Combine(
        ::testing::Values(2, 3, 4, 5), // number of clusters
        ::testing::Values(false, true), // with or without the AUC?
        ::testing::Values(1, 3) // number of threads
    )
);

/*********************************************/

class ScoreMarkersAgainstPairwiseTest : public ::testing::TestWithParam<std::tuple<int, bool> >, public ScoreMarkersTestCore {
protected:
    void SetUp() {
        assemble();
    }

    template<class EffectSummary>
    void compare_effects(int ngroups, const EffectSummary& res, const EffectSummary& other) const {
        for (int s = 0; s < scran::differential_analysis::n_summaries; ++s) {
            EXPECT_EQ(res[s].size(), other[s].size());
            for (int l = 0; l < ngroups; ++l) {
                EXPECT_EQ(res[s][l], other[s][l]);
            }
        }
    }
};

TEST_P(ScoreMarkersAgainstPairwiseTest, Simple) {
    auto param = GetParam();

    auto ngroups = std::get<0>(param);
    std::vector<int> groupings = create_groupings(dense_row->ncol(), ngroups);
    bool do_auc = std::get<1>(param);

    scran::ScoreMarkers chd;
    chd.set_compute_auc(do_auc); // false, if we want to check the running implementations.
    auto res = chd.run(dense_row.get(), groupings.data());

    scran::PairwiseEffects paired;
    paired.set_compute_auc(do_auc); // false, if we want to check the running implementations.
    auto pairres = paired.run(dense_row.get(), groupings.data());

    scran::SummarizeEffects summarizer;
    size_t ngenes = dense_row->nrow();
    auto cohen_summ = summarizer.run(ngenes, ngroups, pairres.cohen.data());
    compare_effects(ngroups, cohen_summ, res.cohen);

    auto lfc_summ = summarizer.run(ngenes, ngroups, pairres.lfc.data());
    compare_effects(ngroups, lfc_summ, res.lfc);

    auto delta_detected_summ = summarizer.run(ngenes, ngroups, pairres.delta_detected.data());
    compare_effects(ngroups, delta_detected_summ, res.delta_detected);

    if (do_auc) {
        auto auc_summ = summarizer.run(ngenes, ngroups, pairres.auc.data());
        compare_effects(ngroups, auc_summ, res.auc);
    }
}

TEST_P(ScoreMarkersAgainstPairwiseTest, Blocked) {
    auto param = GetParam();

    auto combined = tatami::make_DelayedBind<1>(std::vector<std::shared_ptr<tatami::NumericMatrix> >{dense_row, dense_row});
    auto NC = dense_row->ncol();
    auto ngroups = std::get<0>(GetParam());
    std::vector<int> groupings = create_groupings(NC * 2, ngroups);
    std::vector<int> blocks(groupings.size());
    std::fill(blocks.begin() + NC, blocks.end(), 1);

    bool do_auc = std::get<1>(param);
    scran::ScoreMarkers chd;
    chd.set_compute_auc(do_auc); // false, if we want to check the running implementations.
    auto res = chd.run(dense_row.get(), groupings.data());

    scran::PairwiseEffects paired;
    paired.set_compute_auc(do_auc); // false, if we want to check the running implementations.
    auto pairres = paired.run(dense_row.get(), groupings.data());

    scran::SummarizeEffects summarizer;
    size_t ngenes = dense_row->nrow();
    auto cohen_summ = summarizer.run(ngenes, ngroups, pairres.cohen.data());
    compare_effects(ngroups, cohen_summ, res.cohen);

    auto lfc_summ = summarizer.run(ngenes, ngroups, pairres.lfc.data());
    compare_effects(ngroups, lfc_summ, res.lfc);

    auto delta_detected_summ = summarizer.run(ngenes, ngroups, pairres.delta_detected.data());
    compare_effects(ngroups, delta_detected_summ, res.delta_detected);

    if (do_auc) {
        auto auc_summ = summarizer.run(ngenes, ngroups, pairres.auc.data());
        compare_effects(ngroups, auc_summ, res.auc);
    }
}

INSTANTIATE_TEST_CASE_P(
    ScoreMarkersAgainstPairwise,
    ScoreMarkersAgainstPairwiseTest,
    ::testing::Combine(
        ::testing::Values(2, 3, 4, 5), // number of clusters
        ::testing::Values(false, true) // with or without the AUC?
    )
);

/*********************************************/

class ScoreMarkersScenarioTest : public ::testing::TestWithParam<std::tuple<int, bool> >, public ScoreMarkersTestCore {
    void SetUp() {
        assemble();
    }
};

TEST_P(ScoreMarkersScenarioTest, Thresholds) {
    auto ngroups = std::get<0>(GetParam());
    std::vector<int> groupings = create_groupings(dense_row->ncol(), ngroups);

    scran::ScoreMarkers chd;
    bool do_auc = std::get<1>(GetParam());
    chd.set_compute_auc(do_auc); // false, if we want to check the running implementations.
    auto ref = chd.run(dense_row.get(), groupings.data());
    auto out = chd.set_threshold(1).run(dense_row.get(), groupings.data());

    bool some_diff = false;
    for (int s = 0; s <= scran::differential_analysis::MAX; ++s) {
        for (int l = 0; l < ngroups; ++l) {
            for (size_t g = 0; g < dense_row->nrow(); ++g) {
                EXPECT_TRUE(ref.cohen[s][l][g] > out.cohen[s][l][g]);

                if (do_auc) {
                    some_diff |= (ref.auc[s][l][g] > out.auc[s][l][g]);

                    // '>' is not guaranteed due to imprecision with ranks... but (see below).
                    EXPECT_TRUE(ref.auc[s][l][g] >= out.auc[s][l][g]); 
                }
            }
        }
    }

    if (do_auc) {
        EXPECT_TRUE(some_diff); // (from above)... at least one is '>', hopefully.
    }
}

TEST_P(ScoreMarkersScenarioTest, BlockConfounded) {
    auto NC = dense_row->ncol();
    auto ngroups = std::get<0>(GetParam());
    std::vector<int> groupings = create_groupings(NC, ngroups);

    // Block is fully confounded with one group.
    std::vector<int> blocks(NC);
    for (size_t c = 0; c < NC; ++c) {
        blocks[c] = groupings[c] == 0;
    }

    scran::ScoreMarkers chd;
    bool do_auc = std::get<1>(GetParam());
    chd.set_compute_auc(do_auc); // false, if we want to check the running implementations.
    auto comres = chd.run_blocked(dense_row.get(), groupings.data(), blocks.data());

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
    
    // Expect all but the first group to give the same results.
    int ngenes = dense_row->nrow();
    for (int l = 0; l < (ngroups == 2 ? 2 : 1); ++l) {
        for (int s = 0; s < scran::differential_analysis::MIN_RANK; ++s) {
            for (int g = 0; g < ngenes; ++g) {
                EXPECT_TRUE(std::isnan(comres.cohen[s][l][g]));
                EXPECT_TRUE(std::isnan(comres.lfc[s][l][g]));
                EXPECT_TRUE(std::isnan(comres.delta_detected[s][l][g]));
                if (do_auc) {
                    EXPECT_TRUE(std::isnan(comres.auc[s][l][g]));
                }
            }
        }
        for (int g = 0; g < ngenes; ++g) {
            EXPECT_EQ(comres.cohen[scran::differential_analysis::MIN_RANK][l][g], ngenes + 1);
        }
    }

    if (ngroups > 2) {
        for (int l = 1; l < ngroups; ++l) {
            for (int s = 0; s < scran::differential_analysis::n_summaries; ++s) {
                EXPECT_EQ(ref.cohen[s][l - 1], comres.cohen[s][l]);
                EXPECT_EQ(ref.lfc[s][l - 1], comres.lfc[s][l]);
                EXPECT_EQ(ref.delta_detected[s][l - 1], comres.delta_detected[s][l]);
                if (do_auc) {
                    EXPECT_EQ(ref.auc[s][l - 1], comres.auc[s][l]);
                }
            }
        }
    }
}
 
INSTANTIATE_TEST_CASE_P(
    ScoreMarkersScenario,
    ScoreMarkersScenarioTest,
    ::testing::Combine(
        ::testing::Values(2, 3, 4, 5), // number of clusters
        ::testing::Values(false, true) // with or without the AUC?
    )
);

/*********************************************/

TEST(ScoreMarkersTest, MinRank) {
    // Checking that the minimum rank is somewhat sensible,
    // and we didn't feed in the wrong values somewhere.
    int ngenes = 10;
    int nsamples = 8;

    std::vector<double> buffer(ngenes * nsamples, 0);
    for (int i = 0; i < ngenes; ++i) {
        buffer[i * nsamples + 1] = i; // second group gets increasing rank with later genes
        buffer[i * nsamples + 3] = i+1;

        buffer[i * nsamples + 4] = -i; // third group gets decreasing rank with later genes
        buffer[i * nsamples + 6] = -i + 1;
    }

    std::vector<int> grouping{0, 1, 0, 1, 2, 3, 2, 3};
    tatami::DenseRowMatrix<double, int> mat(ngenes, nsamples, std::move(buffer));

    scran::ScoreMarkers chd;
    auto res = chd.run(&mat, grouping.data());

    for (int i = 0; i < ngenes; ++i) {
        EXPECT_EQ(res.cohen[4][1][i], ngenes - i); // second group
        EXPECT_EQ(res.cohen[4][2][i], i + 1);  // third group
    }
}

TEST(ScoreMarkersTest, Options) {
    scran::ScoreMarkers chd;

    chd.set_compute_cohen(false);
    chd.set_compute_cohen(true);
    chd.set_compute_cohen(scran::ScoreMarkers::Defaults::compute_no_summaries());
    chd.set_compute_cohen(scran::ScoreMarkers::Defaults::compute_all_summaries());
    chd.set_compute_cohen(scran::differential_analysis::MIN, true);
    chd.set_compute_cohen(scran::differential_analysis::MIN, false);

    chd.set_compute_auc(false);
    chd.set_compute_auc(true);
    chd.set_compute_auc(scran::ScoreMarkers::Defaults::compute_no_summaries());
    chd.set_compute_auc(scran::ScoreMarkers::Defaults::compute_all_summaries());
    chd.set_compute_auc(scran::differential_analysis::MIN, true);
    chd.set_compute_auc(scran::differential_analysis::MIN, false);

    chd.set_compute_lfc(false);
    chd.set_compute_lfc(true);
    chd.set_compute_lfc(scran::ScoreMarkers::Defaults::compute_no_summaries());
    chd.set_compute_lfc(scran::ScoreMarkers::Defaults::compute_all_summaries());
    chd.set_compute_lfc(scran::differential_analysis::MIN, true);
    chd.set_compute_lfc(scran::differential_analysis::MIN, false);

    chd.set_compute_delta_detected(false);
    chd.set_compute_delta_detected(true);
    chd.set_compute_delta_detected(scran::ScoreMarkers::Defaults::compute_no_summaries());
    chd.set_compute_delta_detected(scran::ScoreMarkers::Defaults::compute_all_summaries());
    chd.set_compute_delta_detected(scran::differential_analysis::MIN, true);
    chd.set_compute_delta_detected(scran::differential_analysis::MIN, false);

    chd.set_summary_min(false);
    chd.set_summary_mean(false);
    chd.set_summary_median(false);
    chd.set_summary_max(false);
    chd.set_summary_min_rank(false);
}
