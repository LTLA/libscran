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

    static ReferenceResults reference(const tatami::NumericMatrix* mat, const unsigned char* features, bool scale) {
        scran::RunPCA runner;
        runner.set_rank(1);
        runner.set_scale(scale);
        auto res = runner.run(mat, features);

        ReferenceResults output;
        output.weights.insert(output.weights.end(), res.rotation.data(), res.rotation.data() + res.rotation.size());
        output.scores.insert(output.scores.end(), res.pcs.data(), res.pcs.data() + res.pcs.size());
        output.prop_var = res.variance_explained[0] / res.total_variance;

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
        ::testing::Values(false, true), // with or without scaling
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

/******************************************************
 ******************************************************/

class ScoreFeatureSetMultiBlockTest : public ::testing::TestWithParam<std::tuple<int, int, bool, int> >, public ScoreFeatureSetTestCore {
protected:
    static std::vector<int> spawn_blocks(int nblocks, int ncells, int seed) {
        std::vector<int> block(ncells);
        std::mt19937_64 rng(seed * 99);
        for (int c = 0; c < ncells; ++c) {
            block[c] = rng() % nblocks;
        }
        return block;
    }

    static std::vector<double> compute_scores(
        const tatami::NumericMatrix* mat, 
        const std::vector<double>& weights, 
        const std::vector<int>& which_features, 
        const std::vector<int>& block,
        const std::vector<std::vector<double> >& centers,
        bool use_scale,
        const std::vector<std::vector<double> >& scales)
    {
        auto ncells = mat->ncol();
        auto nselected = which_features.size();

        std::vector<double> to_add(centers.size());
        for (size_t c = 0; c < centers.size(); ++c) {
            to_add[c] = std::accumulate(centers[c].begin(), centers[c].end(), 0.0);
        }

        std::vector<double> precol(scales.size());
        if (use_scale) {
            for (size_t c = 0; c < scales.size(); ++c) {
                precol[c] = std::inner_product(weights.begin(), weights.end(), scales[c].begin(), 0.0);
            }
        } else {
            double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
            std::fill(precol.begin(), precol.end(), sum);
        }

        std::vector<double> scores(ncells);
        for (int c = 0; c < ncells; ++c) {
            auto col = mat->column(c);
            auto b = block[c];

            for (int s = 0; s < nselected; ++s) {
                auto f = which_features[s];
                scores[c] += weights[s] * (col[f] - centers[b][s]) / (use_scale ? scales[b][s] : 1);
            }

            scores[c] *= precol[b];
            scores[c] += to_add[b];
            scores[c] /= nselected;
        }

        return scores;
    }
};

TEST_P(ScoreFeatureSetMultiBlockTest, Consistency) {
    auto param = GetParam();
    auto seed = std::get<0>(param);
    auto nblocks = std::get<1>(param);
    auto scale = std::get<2>(param);
    auto nthreads = std::get<3>(param);

    int ngenes = 2010;
    int ncells = 151;
    load(ngenes, ncells, seed);
    auto block = spawn_blocks(nblocks, ncells, seed);
    auto features = spawn_features(ngenes, seed);

    scran::ScoreFeatureSet scorer;
    scorer.set_scale(scale);

    auto ref = scorer.run(dense_row.get(), features.data(), block.data());
    EXPECT_EQ(ref.weights.size(), std::accumulate(features.begin(), features.end(), 0));
    EXPECT_EQ(ref.scores.size(), ncells);

    if (nthreads > 1) {
        scorer.set_num_threads(nthreads);
        auto par = scorer.run(dense_row.get(), features.data(), block.data());
        EXPECT_EQ(ref.weights, par.weights);
        EXPECT_EQ(ref.scores, par.scores);
    }

    auto res2 = scorer.run(dense_column.get(), features.data(), block.data());
    compare_almost_equal(ref.weights, res2.weights);
    compare_almost_equal(ref.scores, res2.scores);

    auto res3 = scorer.run(sparse_row.get(), features.data(), block.data());
    compare_almost_equal(ref.weights, res3.weights);
    compare_almost_equal(ref.scores, res3.scores);

    auto res4 = scorer.run(sparse_column.get(), features.data(), block.data());
    compare_almost_equal(ref.weights, res4.weights);
    compare_almost_equal(ref.scores, res4.scores);
}

TEST_P(ScoreFeatureSetMultiBlockTest, Reference) {
    auto param = GetParam();
    auto seed = std::get<0>(param) * 2; // changing the seed a little.
    auto nblocks = std::get<1>(param);
    auto scale = std::get<2>(param);
    auto nthreads = std::get<3>(param);

    int ngenes = 911;
    int ncells = 201;
    load(ngenes, ncells, seed);
    auto block = spawn_blocks(nblocks, ncells, seed);
    auto features = spawn_features(ngenes, seed);

    std::vector<int> which_features;
    for (int f = 0; f < ngenes; ++f) {
        if (features[f]) {
            which_features.push_back(f);
        }
    }
    auto nselected = which_features.size();

    // Reference calculation, with averaging.
    std::vector<ReferenceResults> results;
    std::vector<std::vector<double> > centers;
    std::vector<std::vector<double> > scales;

    results.reserve(nblocks);
    for (int b = 0; b < nblocks; ++b) {
        std::vector<int> subset;
        for (int c = 0; c < ncells; ++c) {
            if (b == block[c]) {
                subset.push_back(c);
            }
        }

        auto sub = tatami::make_DelayedSubset<1>(dense_column, std::move(subset));

        auto all_rs = tatami::row_sums(sub.get());
        auto all_rv = tatami::row_variances(sub.get());
        std::vector<double> sub_rs, sub_rv;
        for (auto f : which_features) {
            sub_rs.push_back(all_rs[f] / sub->ncol());
            sub_rv.push_back(std::sqrt(all_rv[f]));
        }
        centers.push_back(sub_rs);
        scales.push_back(sub_rv);

        results.push_back(reference(sub.get(), features.data(), scale));
        EXPECT_EQ(results.back().weights.size(), nselected);
    }

    std::vector<double> weights(nselected);
    double total_prop = 0;
    for (int b = 0; b < nblocks; ++b) {
        auto proj = std::inner_product(results[b].weights.begin(), results[b].weights.end(), weights.begin(), 0.0);
        double multiplier = (proj < 0 ? -1 : 1) * results[b].prop_var;
        total_prop += results[b].prop_var;
        for (int s = 0; s < nselected; ++s) {
            weights[s] += results[b].weights[s] * multiplier;
        }
    }

    double l2 = 0;
    for (auto& w : weights) {
        w /= total_prop;
        l2 += w * w;
    }

    l2 = std::sqrt(l2);
    for (auto& w : weights) {
        w /= l2;
    }

    // Comparing to our actual thing.
    scran::ScoreFeatureSet scorer;
    scorer.set_num_threads(nthreads);
    scorer.set_scale(scale);

    {
        scorer.set_block_policy(scran::ScoreFeatureSet::BlockPolicy::AVERAGE);
        auto obs = scorer.run(sparse_column.get(), features.data(), block.data());
        compare_almost_equal(obs.weights, weights);

        auto scores = compute_scores(dense_column.get(), weights, which_features, block, centers, scale, scales);
        compare_almost_equal(obs.scores, scores);
    }

    // Maximium also gets a run.
    {
        scorer.set_block_policy(scran::ScoreFeatureSet::BlockPolicy::MAXIMUM);
        auto obs = scorer.run(sparse_row.get(), features.data(), block.data());

        size_t chosen = 0; 
        for (int b = 1; b < nblocks; ++b) {
            if (results[b].prop_var > results[chosen].prop_var) {
                chosen = b;
            }
        }
        compare_almost_equal(obs.weights, results[chosen].weights);

        auto scores = compute_scores(dense_column.get(), obs.weights, which_features, block, centers, scale, scales);
        compare_almost_equal(obs.scores, scores);
    }
}

INSTANTIATE_TEST_CASE_P(
    ScoreFeatureSetMultiBlock,
    ScoreFeatureSetMultiBlockTest,
    ::testing::Combine(
        ::testing::Values(455, 5444), // seeds
        ::testing::Values(1, 2, 3, 4), // number of blocks
        ::testing::Values(false, true), // with or without scaling
        ::testing::Values(1, 3) // number of threads
    )
);

/******************************************************
 ******************************************************/

class ScoreFeatureSetOtherTest : public ::testing::Test, public ScoreFeatureSetTestCore {};

TEST_F(ScoreFeatureSetOtherTest, EdgeCaseGenes) {
    int ngenes = 1011;
    int ncells = 101;
    load(ngenes, ncells, /* seed */ 42);

    scran::ScoreFeatureSet scorer;

    // No features at all.
    {
        std::vector<unsigned char> features(ngenes);
        auto obs = scorer.run(dense_row.get(), features.data());
        EXPECT_TRUE(obs.weights.empty());
        EXPECT_EQ(obs.scores, std::vector<double>(ncells));
    }

    // Single feature.
    {
        std::vector<unsigned char> features(ngenes);
        features[3] = 1;

        auto obs = scorer.run(dense_row.get(), features.data());
        EXPECT_EQ(obs.weights.size(), 1);
        EXPECT_EQ(obs.weights[0], 1);
        EXPECT_EQ(obs.scores, dense_row->row(3));
    }
}

TEST_F(ScoreFeatureSetOtherTest, EdgeCaseBlock) {
    int ngenes = 1011;
    std::vector<unsigned char> features = spawn_features(ngenes, /* seed */ 43);
    size_t nfeatures = std::accumulate(features.begin(), features.end(), 0);

    // Empty dataset.
    {
        load(ngenes, 0, /* seed */ 44);
        scran::ScoreFeatureSet scorer;
        auto obs = scorer.run(dense_row.get(), features.data());
        EXPECT_EQ(obs.weights, std::vector<double>(nfeatures));
        EXPECT_TRUE(obs.scores.empty());
    }

    // 1-length batches lead to an all-zero rotation vector.
    {
        int ncells = 234;
        load(ngenes, ncells, /* seed */ 45);
        std::vector<int> block(ncells);
        std::iota(block.begin(), block.end(), 0);

        scran::ScoreFeatureSet scorer;
        auto obs = scorer.run(dense_row.get(), features.data(), block.data());
        EXPECT_EQ(obs.weights, std::vector<double>(nfeatures));

        // Same behavior in the sparse case.
        auto sobs = scorer.run(sparse_column.get(), features.data(), block.data());
        EXPECT_EQ(sobs.weights, std::vector<double>(nfeatures));
    }

    // Zero length batches are ignored.
    {
        int ncells = 99;
        load(ngenes, ncells, /* seed */ 45);
        std::vector<int> block(ncells, 2); // 0 and 1 are now zero-length.

        scran::ScoreFeatureSet scorer;
        auto obs = scorer.run(dense_column.get(), features.data(), block.data());
        auto ref = scorer.run(dense_column.get(), features.data());
        EXPECT_EQ(obs.weights, ref.weights);
        EXPECT_EQ(obs.scores, ref.scores);
    }
}

TEST_F(ScoreFeatureSetOtherTest, ScoreSanityCheck) {
    int ngenes = 1011;
    int ncells = 101;
    int seed = 9876521;
    load(ngenes, ncells, seed);
    std::vector<unsigned char> features = spawn_features(ngenes, seed);

    // Shifting everything up in one batch. This should manifest as a
    // corresponding shift in the scores for that batch. 
    {
        auto added = tatami::make_DelayedIsometricOp(sparse_column, tatami::DelayedAddScalarHelper(5.6));
        auto combined = tatami::make_DelayedBind<1>(std::vector<std::shared_ptr<tatami::NumericMatrix> >{ sparse_row, added });

        scran::ScoreFeatureSet scorer;
        auto ref = scorer.run(dense_row.get(), features.data());

        std::vector<int> batch(ncells * 2);
        std::fill(batch.begin() + ncells, batch.end(), 1);
        auto obs = scorer.run(combined.get(), features.data(), batch.data());

        compare_almost_equal(ref.weights, obs.weights);
        std::vector<double> first_half(obs.scores.begin(), obs.scores.begin() + ncells);
        compare_almost_equal(first_half, ref.scores);

        std::vector<double> second_half(obs.scores.begin() + ncells, obs.scores.end());
        for (auto& s : second_half) { s -= 5.6; }
        compare_almost_equal(second_half, first_half);
    }

    // Scaling everything up in one batch. This should manifest as a
    // corresponding scaling in the scores for that batch, regardless
    // of whether set_scale is true or not.
    {
        auto scaled = tatami::make_DelayedIsometricOp(sparse_column, tatami::DelayedMultiplyScalarHelper(1.5));
        auto combined = tatami::make_DelayedBind<1>(std::vector<std::shared_ptr<tatami::NumericMatrix> >{ sparse_row, scaled });
        std::vector<int> batch(ncells * 2);
        std::fill(batch.begin() + ncells, batch.end(), 1);

        // Without scaling.
        scran::ScoreFeatureSet scorer;
        {
            auto ref = scorer.run(dense_row.get(), features.data());
            auto obs = scorer.run(combined.get(), features.data(), batch.data());

            compare_almost_equal(ref.weights, obs.weights);
            std::vector<double> first_half(obs.scores.begin(), obs.scores.begin() + ncells);
            compare_almost_equal(first_half, ref.scores);

            std::vector<double> second_half(obs.scores.begin() + ncells, obs.scores.end());
            for (auto& s : second_half) { s /= 1.5; }
            compare_almost_equal(second_half, first_half);
        }

        // Again, with scaling.
        scorer.set_scale(true);
        {
            auto ref = scorer.run(dense_row.get(), features.data());
            auto obs = scorer.run(combined.get(), features.data(), batch.data());

            compare_almost_equal(ref.weights, obs.weights);
            std::vector<double> first_half(obs.scores.begin(), obs.scores.begin() + ncells);
            compare_almost_equal(first_half, ref.scores);

            std::vector<double> second_half(obs.scores.begin() + ncells, obs.scores.end());
            for (auto& s : second_half) { s /= 1.5; }
            compare_almost_equal(second_half, first_half);
        }
    }
}

