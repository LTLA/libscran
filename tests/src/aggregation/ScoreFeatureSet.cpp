#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "../utils/compare_almost_equal.h"
#include "../data/Simulator.hpp"
#include "scran/aggregation/ScoreFeatureSet.hpp"

class ScoreFeatureSetTest : public ::testing::TestWithParam<std::tuple<int, scran::ScoreFeatureSet::BlockPolicy, int> > {
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
};

TEST_P(ScoreFeatureSetTest, SingleBlockConsistency) {
    auto param = GetParam();
    auto seed = std::get<0>(param);
    auto policy = std::get<1>(param);
    auto nthreads = std::get<2>(param);

    int ngenes = 1011;
    int ncells = 101;
    load(ngenes, ncells, seed);

    std::vector<unsigned char> features(ngenes);
    std::mt19937_64 rng(seed * 10);
    for (int g = 0; g < ngenes; ++g) {
        features[g] = (rng() % 1000) < 100; // 10% chance, roughly, of being in the feature set.
    }

    scran::ScoreFeatureSet scorer;
    scorer.set_block_policy(policy);

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

INSTANTIATE_TEST_CASE_P(
    ScoreFeatureSet,
    ScoreFeatureSetTest,
    ::testing::Combine(
        ::testing::Values(1999, 28888, 377777), // seeds
        ::testing::Values(scran::ScoreFeatureSet::BlockPolicy::AVERAGE, scran::ScoreFeatureSet::BlockPolicy::MAXIMUM), // policy
        ::testing::Values(1, 3) // number of threads
    )
);


