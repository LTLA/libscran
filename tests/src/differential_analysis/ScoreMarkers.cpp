#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "tatami/tatami.hpp"
#include "scran/differential_analysis/ScoreMarkers.hpp"
#include "scran/differential_analysis/PairwiseEffects.hpp"
#include "scran/differential_analysis/SummarizeEffects.hpp"

#include "../utils/compare_almost_equal.h"
#include "utils.h"

class ScoreMarkersTestCore : public DifferentialAnalysisTestCore {
protected:
    template<class Left, class Right>
    static void compare_basic(const Left& res, const Right& other, int ngroups, int nblocks) {
        EXPECT_EQ(other.means.size(), ngroups);
        EXPECT_EQ(other.detected.size(), ngroups);
        EXPECT_EQ(res.means.size(), ngroups);
        EXPECT_EQ(res.detected.size(), ngroups);

        for (int l = 0; l < ngroups; ++l) {
            compare_almost_equal(res.means[l], other.means[l]);
            compare_almost_equal(res.detected[l], other.detected[l]);
        }
    }

    template<class EffectVectors>
    static void compare_effects(int ngroups, const EffectVectors& res, const EffectVectors& other) {
        EXPECT_EQ(res.size(), scran::differential_analysis::n_summaries);
        EXPECT_EQ(other.size(), scran::differential_analysis::n_summaries);

        // Don't compare min-rank here, as minor numerical differences
        // can change the ranks by a small amount when effects are tied.
        for (int s = 0; s < scran::differential_analysis::MIN_RANK; ++s) {
            EXPECT_EQ(res[s].size(), ngroups);
            EXPECT_EQ(other[s].size(), ngroups);
            for (int l = 0; l < ngroups; ++l) {
                compare_almost_equal(res[s][l], other[s][l]);
            }
        }
    }

    template<class Result>
    static void compare_effects(int ngroups, const Result& res, const Result& other, bool do_auc) {
        compare_effects(ngroups, res.cohen, other.cohen);
        compare_effects(ngroups, res.lfc, other.lfc);
        compare_effects(ngroups, res.delta_detected, other.delta_detected);
        if (do_auc) {
            compare_effects(ngroups, res.auc, other.auc);
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

/*********************************************/

class ScoreMarkersTest : public ScoreMarkersTestCore, public ::testing::TestWithParam<std::tuple<int, bool, int> > {
protected:
    void SetUp() {
        assemble();
    }
};

TEST_P(ScoreMarkersTest, Basic) {
    auto param = GetParam();
    auto ngroups = std::get<0>(param);
    bool do_auc = std::get<1>(param);
    auto nthreads = std::get<2>(param);

    std::vector<int> groupings = create_groupings(dense_row->ncol(), ngroups);
    size_t ngenes = dense_row->nrow();

    scran::ScoreMarkers chd;
    chd.set_compute_auc(do_auc); // false, if we want to check the running implementations.
    chd.set_num_threads(nthreads);

    // Avoid issues with numerical precision when the weights are different,
    // even if they ultimately cancel out in the calculation.
    chd.set_block_weight_policy(scran::WeightPolicy::NONE);

    auto res = chd.run(dense_row.get(), groupings.data());

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

    if (nthreads > 1) {
        // Comparing to the same call, but not parallelized.
        chd.set_num_threads(1);
        auto res1 = chd.run(dense_row.get(), groupings.data());
        compare_basic(res, res1, ngroups, 1);
        compare_effects(ngroups, res, res1, do_auc);
    }
}

// Comparing the efficient ScoreMarkers implementation against the
// PairwiseEffects + SummarizeEffects combination, which is less mind-bending
// but requires holding a large 3D matrix in memory.

TEST_P(ScoreMarkersTest, AgainstPairwise) {
    auto param = GetParam();
    auto ngroups = std::get<0>(param);
    bool do_auc = std::get<1>(param);
    auto nthreads = std::get<2>(param);

    std::vector<int> groupings = create_groupings(dense_row->ncol(), ngroups);
    scran::ScoreMarkers chd;
    chd.set_num_threads(nthreads);
    chd.set_compute_auc(do_auc); // false, if we want to check the running implementations.
    auto res = chd.run(dense_row.get(), groupings.data());

    scran::PairwiseEffects paired;
    paired.set_num_threads(nthreads);
    paired.set_compute_auc(do_auc); // false, if we want to check the running implementations.
    auto pairres = paired.run(dense_row.get(), groupings.data());
    compare_basic(res, pairres, ngroups, 1);

    scran::SummarizeEffects summarizer;
    summarizer.set_num_threads(nthreads);
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

INSTANTIATE_TEST_SUITE_P(
    ScoreMarkers,
    ScoreMarkersTest,
    ::testing::Combine(
        ::testing::Values(2, 3, 4, 5), // number of clusters
        ::testing::Values(false, true), // with or without the AUC?
        ::testing::Values(1, 3) // number of threads
    )
);

/*********************************************/

class ScoreMarkersBlockedTest : public ScoreMarkersTestCore, public ::testing::TestWithParam<std::tuple<int, bool, scran::WeightPolicy, int> > {
protected:
    void SetUp() {
        assemble();
    }
};

TEST_P(ScoreMarkersBlockedTest, AgainstPairwise) {
    auto param = GetParam();
    auto ngroups = std::get<0>(param);
    bool do_auc = std::get<1>(param);
    auto policy = std::get<2>(param);
    auto nthreads = std::get<3>(param);

    auto NC = dense_row->ncol();
    std::vector<int> groupings = create_groupings(NC, ngroups);
    std::vector<int> blocks = create_blocks(NC, 3);

    scran::ScoreMarkers chd;
    chd.set_compute_auc(do_auc); // false, if we want to check the running implementations.
    chd.set_num_threads(nthreads);
    chd.set_block_weight_policy(policy);
    chd.set_variable_block_weight_parameters({0, 1000}); // for some more interesting variable block weights.
    auto res = chd.run_blocked(dense_row.get(), groupings.data(), blocks.data());

    scran::PairwiseEffects paired;
    paired.set_compute_auc(do_auc); // false, if we want to check the running implementations.
    paired.set_num_threads(nthreads);
    paired.set_block_weight_policy(policy);
    paired.set_variable_block_weight_parameters({0, 1000});
    auto pairres = paired.run_blocked(dense_row.get(), groupings.data(), blocks.data());
    compare_basic(res, pairres, ngroups, 2);

    scran::SummarizeEffects summarizer;
    summarizer.set_num_threads(nthreads);

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

INSTANTIATE_TEST_SUITE_P(
    ScoreMarkers,
    ScoreMarkersBlockedTest,
    ::testing::Combine(
        ::testing::Values(2, 3, 4, 5), // number of clusters
        ::testing::Values(false, true), // with or without the AUC?
        ::testing::Values(scran::WeightPolicy::NONE, scran::WeightPolicy::EQUAL, scran::WeightPolicy::VARIABLE), // block weighting method.
        ::testing::Values(1, 3) // number of threads
    )
);

/*********************************************/

// Checking that we get the same results for different cache sizes.

class ScoreMarkersCacheTest : 
    public ::testing::TestWithParam<std::tuple<int, int> >, 
    public DifferentialAnalysisTestCore {
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

TEST_P(ScoreMarkersCacheTest, Basic) {
    auto param = GetParam();

    auto ngroups = std::get<0>(param);
    std::vector<int> groupings = create_groupings(dense_row->ncol(), ngroups);
    auto cache_size = std::get<1>(param);

    scran::ScoreMarkers chd;
    chd.set_cache_size(cache_size);
    auto cached = chd.run(dense_row.get(), groupings.data());
    chd.set_cache_size(0);
    auto uncached = chd.run(dense_row.get(), groupings.data());

    compare_effects(ngroups, cached.cohen, uncached.cohen);
    compare_effects(ngroups, cached.auc, uncached.auc);
    compare_effects(ngroups, cached.lfc, uncached.lfc);
    compare_effects(ngroups, cached.delta_detected, uncached.delta_detected);
}

INSTANTIATE_TEST_SUITE_P(
    ScoreMarkersCache,
    ScoreMarkersCacheTest,
    ::testing::Combine(
        ::testing::Values(2, 4, 8, 16), // number of clusters
        ::testing::Values(5, 10, 20, 100) // size of the cache
    )
);

/*********************************************/

class ScoreMarkersScenarioTest : public ::testing::Test, public DifferentialAnalysisTestCore {
protected:
    void SetUp() {
        assemble();
    }
};

TEST_F(ScoreMarkersScenarioTest, Thresholds) {
    int ngroups = 3;
    std::vector<int> groupings = create_groupings(dense_row->ncol(), ngroups);

    scran::ScoreMarkers chd;
    auto ref = chd.run(dense_row.get(), groupings.data());
    auto out = chd.set_threshold(1).run(dense_row.get(), groupings.data());

    bool some_diff = false;
    for (int s = 0; s <= scran::differential_analysis::MAX; ++s) {
        for (int l = 0; l < ngroups; ++l) {
            for (size_t g = 0; g < dense_row->nrow(); ++g) {
                EXPECT_TRUE(ref.cohen[s][l][g] > out.cohen[s][l][g]);

                // '>' is not guaranteed due to imprecision with ranks... but (see below).
                some_diff |= (ref.auc[s][l][g] > out.auc[s][l][g]);
                EXPECT_TRUE(ref.auc[s][l][g] >= out.auc[s][l][g]); 
            }
        }
    }

    EXPECT_TRUE(some_diff); // (from above)... at least one is '>', hopefully.
}

TEST_F(ScoreMarkersScenarioTest, BlockConfounded) {
    auto NC = dense_row->ncol();
    int ngroups = 4;
    std::vector<int> groupings = create_groupings(NC, ngroups);

    // Block is fully confounded with one group.
    std::vector<int> blocks(NC);
    int lost = 0;
    for (size_t c = 0; c < NC; ++c) {
        blocks[c] = groupings[c] == lost;
    }

    scran::ScoreMarkers chd;
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

    // First group is effectively all-NA.
    int ngenes = dense_row->nrow();
    for (int s = 0; s < scran::differential_analysis::MIN_RANK; ++s) {
        for (int g = 0; g < ngenes; ++g) {
            EXPECT_TRUE(std::isnan(comres.cohen[s][lost][g]));
            EXPECT_TRUE(std::isnan(comres.lfc[s][lost][g]));
            EXPECT_TRUE(std::isnan(comres.delta_detected[s][lost][g]));
            EXPECT_TRUE(std::isnan(comres.auc[s][lost][g]));
        }
    }
    for (int g = 0; g < ngenes; ++g) {
        EXPECT_EQ(comres.cohen[scran::differential_analysis::MIN_RANK][lost][g], ngenes + 1);
    }

    // Expect all but the first group to give the same results.
    for (int l = 0; l < ngroups; ++l) {
        if (l == lost) {
            continue;
        }
        for (int s = 0; s < scran::differential_analysis::n_summaries; ++s) {
            EXPECT_EQ(ref.cohen[s][l - 1], comres.cohen[s][l]);
            EXPECT_EQ(ref.lfc[s][l - 1], comres.lfc[s][l]);
            EXPECT_EQ(ref.delta_detected[s][l - 1], comres.delta_detected[s][l]);
            EXPECT_EQ(ref.auc[s][l - 1], comres.auc[s][l]);
        }
    }
}

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
