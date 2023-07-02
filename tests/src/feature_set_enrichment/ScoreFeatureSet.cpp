#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "../utils/compare_almost_equal.h"
#include "../data/Simulator.hpp"

#include "scran/feature_set_enrichment/ScoreFeatureSet.hpp"
#include "scran/dimensionality_reduction/SimplePca.hpp"
#include "aarand/aarand.hpp"

class ScoreFeatureSetTestCore {
protected:
    std::shared_ptr<tatami::NumericMatrix> sparse_column, sparse_row, dense_row, dense_column;

    void load(int nr, int nc, int seed) {
        Simulator sim;
        sim.seed = seed;
        auto mat = sim.matrix(nr, nc);
        dense_row.reset(new tatami::DenseRowMatrix<double, int>(std::move(mat)));
        dense_column = tatami::convert_to_dense(dense_row.get(), 1);
        sparse_row = tatami::convert_to_sparse(dense_row.get(), 0);
        sparse_column = tatami::convert_to_sparse(dense_row.get(), 1);
    }

    static std::vector<unsigned char> spawn_features(int ngenes, int seed) {
        std::vector<unsigned char> features(ngenes);
        std::mt19937_64 rng(seed * 10);
        for (int g = 0; g < ngenes; ++g) {
            features[g] = (rng() % 1000) < 100; // 10% chance, roughly, of being in the feature set.
        }
        return features;
    }

    struct ReferenceResults {
        std::vector<double> weights, scores;
    };

    static ReferenceResults reference(const tatami::NumericMatrix* mat, const unsigned char* features, bool scale) {
        scran::SimplePca runner;
        runner.set_rank(1);
        runner.set_scale(scale);
        runner.set_return_rotation(true);
        auto res = runner.run(mat, features);

        ReferenceResults output;
        output.weights.insert(output.weights.end(), res.rotation.data(), res.rotation.data() + res.rotation.size());
        output.scores.insert(output.scores.end(), res.pcs.data(), res.pcs.data() + res.pcs.size());

        // Adjusting the scores to convert them into low-rank colsums.
        auto rowsums = tatami::row_sums(mat);
        auto rowvars = tatami::row_variances(mat);
        double precol = 0;
        double to_add = 0;
        int fcount = 0;
        for (size_t r = 0; r < mat->nrow(); ++r) {
            if (features[r]) {
                to_add += rowsums[r] / mat->ncol();
                precol += (scale ? std::sqrt(rowvars[r]) : 1) * output.weights[fcount];
                ++fcount;
            }
        }

        for (auto& x : output.scores) {
            x *= precol;
            x += to_add;
            x /= fcount;
        }

        return output;
    }
};

/******************************************************
 ******************************************************/

class ScoreFeatureSetSingleBlockTest : public ::testing::TestWithParam<std::tuple<int, bool, int> >, public ScoreFeatureSetTestCore {};

TEST_P(ScoreFeatureSetSingleBlockTest, Consistency) {
    auto param = GetParam();
    auto seed = std::get<0>(param);
    auto scale = std::get<1>(param);
    auto nthreads = std::get<2>(param);

    int ngenes = 1011;
    int ncells = 101;
    load(ngenes, ncells, seed);

    std::vector<unsigned char> features = spawn_features(ngenes, seed);
    scran::ScoreFeatureSet scorer;
    scorer.set_scale(scale);

    auto ref = scorer.run(dense_row.get(), features.data());
    EXPECT_EQ(ref.weights.size(), std::accumulate(features.begin(), features.end(), 0));
    EXPECT_EQ(ref.scores.size(), ncells);

    if (nthreads > 1) {
        scorer.set_num_threads(nthreads);
        auto par = scorer.run(dense_row.get(), features.data());
        EXPECT_EQ(ref.weights, par.weights);
        EXPECT_EQ(ref.scores, par.scores);
    }

    auto res2 = scorer.run(dense_column.get(), features.data());
    compare_almost_equal(ref.weights, res2.weights);
    compare_almost_equal(ref.scores, res2.scores);

    auto res3 = scorer.run(sparse_row.get(), features.data());
    compare_almost_equal(ref.weights, res3.weights);
    compare_almost_equal(ref.scores, res3.scores);

    auto res4 = scorer.run(sparse_column.get(), features.data());
    compare_almost_equal(ref.weights, res4.weights);
    compare_almost_equal(ref.scores, res4.scores);
}

TEST_P(ScoreFeatureSetSingleBlockTest, Reference) {
    auto param = GetParam();
    auto seed = std::get<0>(param) * 2; // changing the seed a little.
    auto scale = std::get<1>(param);
    auto nthreads = std::get<2>(param);

    int ngenes = 1211;
    int ncells = 101;
    load(ngenes, ncells, seed);

    std::vector<unsigned char> features = spawn_features(ngenes, seed);
    auto ref = reference(sparse_column.get(), features.data(), scale);

    scran::ScoreFeatureSet scorer;
    scorer.set_num_threads(nthreads);
    scorer.set_scale(scale);

    auto obs = scorer.run(sparse_column.get(), features.data());
    compare_almost_equal(obs.weights, ref.weights);
    compare_almost_equal(obs.scores, ref.scores);

    // Trying in blocking mode; this should yield the same results.
    std::vector<int> block(ncells);
    auto obsb = scorer.run_blocked(sparse_column.get(), features.data(), block.data());
    compare_almost_equal(obsb.weights, ref.weights);
    compare_almost_equal(obsb.scores, ref.scores);

    scorer.set_weight_policy(scran::ScoreFeatureSet::WeightPolicy::NONE);
    auto obsb2 = scorer.run_blocked(sparse_column.get(), features.data(), block.data());
    compare_almost_equal(obsb2.weights, ref.weights);
    compare_almost_equal(obsb2.scores, ref.scores);
}

INSTANTIATE_TEST_CASE_P(
    ScoreFeatureSetSingleBlock,
    ScoreFeatureSetSingleBlockTest,
    ::testing::Combine(
        ::testing::Values(1999, 28888, 377777), // seeds
        ::testing::Values(false, true), // with or without scaling
        ::testing::Values(1, 3) // number of threads
    )
);

///******************************************************
// ******************************************************/
//
//class ScoreFeatureSetOtherTest : public ::testing::Test, public ScoreFeatureSetTestCore {};
//
//TEST_F(ScoreFeatureSetOtherTest, EdgeCaseGenes) {
//    int ngenes = 1011;
//    int ncells = 101;
//    load(ngenes, ncells, /* seed */ 42);
//
//    scran::ScoreFeatureSet scorer;
//
//    // No features at all.
//    {
//        std::vector<unsigned char> features(ngenes);
//        auto obs = scorer.run(dense_row.get(), features.data());
//        EXPECT_TRUE(obs.weights.empty());
//        EXPECT_EQ(obs.scores, std::vector<double>(ncells));
//    }
//
//    // Single feature.
//    {
//        std::vector<unsigned char> features(ngenes);
//        features[3] = 1;
//
//        auto obs = scorer.run(dense_row.get(), features.data());
//        EXPECT_EQ(obs.weights.size(), 1);
//        EXPECT_EQ(obs.weights[0], 1);
//        EXPECT_EQ(obs.scores, dense_row->dense_row()->fetch(3));
//    }
//}
//
//TEST_F(ScoreFeatureSetOtherTest, EdgeCaseBlock) {
//    int ngenes = 1011;
//    std::vector<unsigned char> features = spawn_features(ngenes, /* seed */ 43);
//    size_t nfeatures = std::accumulate(features.begin(), features.end(), 0);
//
//    // Empty dataset.
//    {
//        load(ngenes, 0, /* seed */ 44);
//        scran::ScoreFeatureSet scorer;
//        auto obs = scorer.run(dense_row.get(), features.data());
//        EXPECT_EQ(obs.weights, std::vector<double>(nfeatures));
//        EXPECT_TRUE(obs.scores.empty());
//    }
//
//    // Zero length batches are ignored.
//    {
//        int ncells = 99;
//        load(ngenes, ncells, /* seed */ 45);
//        std::vector<int> block(ncells, 2); // 0 and 1 are now zero-length.
//
//        scran::ScoreFeatureSet scorer;
//        auto obs = scorer.run_blocked(dense_column.get(), features.data(), block.data());
//        auto ref = scorer.run(dense_column.get(), features.data());
//        EXPECT_EQ(obs.weights, ref.weights);
//        EXPECT_EQ(obs.scores, ref.scores);
//    }
//}
//
//TEST_F(ScoreFeatureSetOtherTest, ScoreSanityCheck) {
//    int ngenes = 1011;
//    int ncells = 101;
//    int seed = 9876521;
//    load(ngenes, ncells, seed);
//    std::vector<unsigned char> features = spawn_features(ngenes, seed);
//
//    // Shifting everything up in one batch. This should manifest as a
//    // corresponding shift in the scores for that batch. 
//    {
//        auto added = tatami::make_DelayedUnaryIsometricOp(sparse_column, tatami::make_DelayedAddScalarHelper(5.6));
//        auto combined = tatami::make_DelayedBind<1>(std::vector<std::shared_ptr<tatami::NumericMatrix> >{ sparse_row, added });
//
//        scran::ScoreFeatureSet scorer;
//        auto ref = scorer.run(dense_row.get(), features.data());
//
//        std::vector<int> batch(ncells * 2);
//        std::fill(batch.begin() + ncells, batch.end(), 1);
//        auto obs = scorer.run_blocked(combined.get(), features.data(), batch.data());
//
//        compare_almost_equal(ref.weights, obs.weights);
//        std::vector<double> first_half(obs.scores.begin(), obs.scores.begin() + ncells);
//        compare_almost_equal(first_half, ref.scores);
//
//        std::vector<double> second_half(obs.scores.begin() + ncells, obs.scores.end());
//        for (auto& s : second_half) { s -= 5.6; }
//        compare_almost_equal(second_half, first_half);
//    }
//
//    // Scaling everything up in one batch. This should manifest as a
//    // corresponding scaling in the scores for that batch, regardless
//    // of whether set_scale is true or not.
//    {
//        auto scaled = tatami::make_DelayedUnaryIsometricOp(sparse_column, tatami::make_DelayedMultiplyScalarHelper(1.5));
//        auto combined = tatami::make_DelayedBind<1>(std::vector<std::shared_ptr<tatami::NumericMatrix> >{ sparse_row, scaled });
//        std::vector<int> batch(ncells * 2);
//        std::fill(batch.begin() + ncells, batch.end(), 1);
//
//        // Without scaling.
//        scran::ScoreFeatureSet scorer;
//        {
//            auto ref = scorer.run(dense_row.get(), features.data());
//            auto obs = scorer.run_blocked(combined.get(), features.data(), batch.data());
//
//            compare_almost_equal(ref.weights, obs.weights);
//            std::vector<double> first_half(obs.scores.begin(), obs.scores.begin() + ncells);
//            compare_almost_equal(first_half, ref.scores);
//
//            std::vector<double> second_half(obs.scores.begin() + ncells, obs.scores.end());
//            for (auto& s : second_half) { s /= 1.5; }
//            compare_almost_equal(second_half, first_half);
//        }
//
//        // Again, with scaling.
//        scorer.set_scale(true);
//        {
//            auto ref = scorer.run(dense_row.get(), features.data());
//            auto obs = scorer.run_blocked(combined.get(), features.data(), batch.data());
//
//            compare_almost_equal(ref.weights, obs.weights);
//            std::vector<double> first_half(obs.scores.begin(), obs.scores.begin() + ncells);
//            compare_almost_equal(first_half, ref.scores);
//
//            std::vector<double> second_half(obs.scores.begin() + ncells, obs.scores.end());
//            for (auto& s : second_half) { s /= 1.5; }
//            compare_almost_equal(second_half, first_half);
//        }
//    }
//}
//
