#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "../utils/compare_almost_equal.h"
#include "../data/Simulator.hpp"

#include "scran/aggregation/ScoreFeatureSet.hpp"
#include "scran/dimensionality_reduction/RunPCA.hpp"
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
        double prop_var;
    };

    static ReferenceResults reference(const tatami::NumericMatrix* mat, const unsigned char* features) {
        scran::RunPCA runner;
        runner.set_rank(1);
        auto res = runner.run(mat, features);

        ReferenceResults output;
        output.weights.insert(output.weights.end(), res.rotation.data(), res.rotation.data() + res.rotation.size());
        output.scores.insert(output.scores.end(), res.pcs.data(), res.pcs.data() + res.pcs.size());

        auto precol = std::accumulate(output.weights.begin(), output.weights.end(), 0.0);
        for (auto& x : output.scores) {
            x *= precol;
        }

        output.prop_var = res.variance_explained[0] / res.total_variance;
        return output;
    }
};

/******************************************************
 ******************************************************/

class ScoreFeatureSetSingleBlockTest : public ::testing::TestWithParam<std::tuple<int, int> >, public ScoreFeatureSetTestCore {};

TEST_P(ScoreFeatureSetSingleBlockTest, Consistency) {
    auto param = GetParam();
    auto seed = std::get<0>(param);
    auto nthreads = std::get<1>(param);

    int ngenes = 1011;
    int ncells = 101;
    load(ngenes, ncells, seed);

    std::vector<unsigned char> features = spawn_features(ngenes, seed);
    scran::ScoreFeatureSet scorer;

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
    auto nthreads = std::get<1>(param);

    int ngenes = 1011;
    int ncells = 101;
    load(ngenes, ncells, seed);

    std::vector<unsigned char> features = spawn_features(ngenes, seed);
    auto ref = reference(sparse_column.get(), features.data());

    scran::ScoreFeatureSet scorer;
    scorer.set_num_threads(nthreads);

    {
        scorer.set_block_policy(scran::ScoreFeatureSet::BlockPolicy::AVERAGE);
        auto obs = scorer.run(sparse_column.get(), features.data());
        compare_almost_equal(obs.weights, ref.weights);
        compare_almost_equal(obs.scores, ref.scores);
    }

    {
        scorer.set_block_policy(scran::ScoreFeatureSet::BlockPolicy::MAXIMUM);
        auto obs = scorer.run(sparse_column.get(), features.data());
        compare_almost_equal(obs.weights, ref.weights);
        compare_almost_equal(obs.scores, ref.scores);
    }
}

INSTANTIATE_TEST_CASE_P(
    ScoreFeatureSetSingleBlock,
    ScoreFeatureSetSingleBlockTest,
    ::testing::Combine(
        ::testing::Values(1999, 28888, 377777), // seeds
        ::testing::Values(1, 3) // number of threads
    )
);

/******************************************************
 ******************************************************/

class CombineRotationVectorsTest : public ::testing::Test {
protected:
    struct Exposed : public scran::ScoreFeatureSet {
        std::vector<double> run(const std::vector<Eigen::MatrixXd>& rotation, const std::vector<double>& variance_explained) const {
            return this->combine_rotation_vectors(rotation, variance_explained);
        }
    };

    std::vector<Eigen::MatrixXd> rotation;

    void fill(size_t nblocks, size_t nfeatures, int seed) {
        std::mt19937_64 rng(seed);
        rotation.clear();
        rotation.reserve(nblocks);

        for (size_t i = 0; i < nblocks; ++i) {
            rotation.emplace_back(nfeatures, 1);

            double l2 = 0;
            for (size_t f = 0; f < nfeatures; ++f) {
                auto res = aarand::standard_normal(rng);
                rotation.back()(f, 0) = res.first;
                l2 += res.first * res.first;
            }

            l2 = std::sqrt(l2);
            if (l2) {
                for (size_t f = 0; f < nfeatures; ++f) {
                    rotation.back()(f, 0) /= l2;
                }
            }
        }
    }
};

TEST_F(CombineRotationVectorsTest, Average) {
    Exposed exp;

    // Identical rotation vectors.
    {
        size_t nfeatures = 111;
        fill(1, nfeatures, /*seed*/ 9999);

        std::vector<double> variance_explained { 0.1, 0.9 };
        rotation.push_back(rotation[0]);
        auto observed = exp.run(rotation, variance_explained);

        std::vector<double> expected(rotation[0].data(), rotation[0].data() + nfeatures);
        compare_almost_equal(observed, expected);
    }

    // Two flipped rotation vectors; the function should flip it back to the first.
    {
        size_t nfeatures = 131;
        fill(1, nfeatures, /*seed*/ 9999);

        std::vector<double> variance_explained { 0.9, 0.9 };
        rotation.emplace_back(-rotation[0]);

        // Check that it's zeroed.
        EXPECT_EQ((rotation[0] + rotation[1]).norm(), 0);

        auto observed = exp.run(rotation, variance_explained);
        std::vector<double> expected(rotation[0].data(), rotation[0].data() + nfeatures);
        compare_almost_equal(observed, expected);
    }

    // 2 different rotation vectors.
    {
        size_t nfeatures = 100;
        std::vector<double> variance_explained { 0.55, 0.75 };
        fill(variance_explained.size(), nfeatures, /*seed*/ 123123);

        auto proj = rotation[0].col(0).dot(rotation[1].col(0));
        Eigen::VectorXd raw_expected = rotation[0] * variance_explained[0] + rotation[1] * variance_explained[1] * (proj > 0 ? 1 : -1);
        raw_expected /= std::accumulate(variance_explained.begin(), variance_explained.end(), 0.0);
        raw_expected /= raw_expected.norm();

        auto observed = exp.run(rotation, variance_explained);
        std::vector<double> expected(raw_expected.data(), raw_expected.data() + raw_expected.size());
        compare_almost_equal(observed, expected);
    }

    // Three rotation vectors.
    {
        size_t nfeatures = 99;
        std::vector<double> variance_explained { 0.8, 0.2, 0.2 };
        fill(variance_explained.size() - 1, nfeatures, /*seed*/ 456456456);

        // Orthogonalize the second vector w.r.t. the first.
        auto proj = rotation[1].col(0).dot(rotation[0].col(0));
        rotation[1].col(0) -= proj * rotation[0].col(0);
        rotation[1].col(0) /= rotation[1].col(0).norm();

        EXPECT_FLOAT_EQ(rotation[1].col(0).norm(), 1);
        EXPECT_TRUE(std::abs(rotation[0].col(0).dot(rotation[1].col(0))) < 1e-8);

        // Adding another one, but flipped.
        rotation.emplace_back(-rotation[1]);
        auto observed = exp.run(rotation, variance_explained);

        // Computing the expected value.
        rotation.pop_back();
        auto ref = exp.run(rotation, { 0.8, 0.4 });
        compare_almost_equal(observed, ref);
    }
}

TEST_F(CombineRotationVectorsTest, AverageWithZeros) {
    Exposed exp;

    // Skip blocks with zero variance explained.
    {
        size_t nfeatures = 151;
        std::vector<double> variance_explained { 0., 0.2, 0. };
        fill(variance_explained.size(), nfeatures, /*seed*/ 456456456);

        auto observed = exp.run(rotation, variance_explained);
        std::vector<double> expected(rotation[1].data(), rotation[1].data() + nfeatures);
        compare_almost_equal(observed, expected);
    }

    // Total variance explained is zero.
    {
        size_t nfeatures = 87;
        std::vector<double> variance_explained { 0., 0. };
        fill(variance_explained.size(), nfeatures, /*seed*/ 1357);

        auto observed = exp.run(rotation, variance_explained);
        std::vector<double> expected(nfeatures);
        compare_almost_equal(observed, expected);
    }

    // L2 norm is zero.
    {
        size_t nfeatures = 77;
        std::vector<double> variance_explained { 0.1, 0.2 };
        rotation.clear();
        for (size_t i = 0; i < 2; ++i) {
            rotation.emplace_back(nfeatures, 1);
            rotation.back().fill(0);
        }

        auto observed = exp.run(rotation, variance_explained);
        std::vector<double> expected(nfeatures);
        compare_almost_equal(observed, expected);
    }
}

TEST_F(CombineRotationVectorsTest, Maximum) {
    Exposed exp;
    exp.set_block_policy(scran::ScoreFeatureSet::BlockPolicy::MAXIMUM);

    {
        size_t nfeatures = 111;
        std::vector<double> variance_explained { 0.1, 0.9 };
        fill(variance_explained.size(), nfeatures, /*seed*/ 9999);

        auto observed = exp.run(rotation, variance_explained);
        std::vector<double> expected(rotation[1].data(), rotation[1].data() + nfeatures);
        compare_almost_equal(observed, expected);
    }

    {
        size_t nfeatures = 112;
        std::vector<double> variance_explained { 0.1, 0.05, 0.9 };
        fill(variance_explained.size(), nfeatures, /*seed*/ 8888);

        auto observed = exp.run(rotation, variance_explained);
        std::vector<double> expected(rotation[2].data(), rotation[2].data() + nfeatures);
        compare_almost_equal(observed, expected);
    }
}
