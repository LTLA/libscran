#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "../data/data.h"
#include "../utils/compare_almost_equal.h"

#include "tatami/tatami.hpp"

#include "scran/feature_selection/ModelGeneVariances.hpp"

#include <cmath>

class ModelGeneVariancesTest : public ::testing::TestWithParam<int> {
protected:
    void SetUp() {
        dense_row = std::unique_ptr<tatami::NumericMatrix>(new tatami::DenseRowMatrix<double>(sparse_nrow, sparse_ncol, sparse_matrix));
        dense_column = tatami::convert_to_dense(dense_row.get(), 1);
        sparse_row = tatami::convert_to_sparse(dense_row.get(), 0);
        sparse_column = tatami::convert_to_sparse(dense_row.get(), 1);
    }
protected:
    std::shared_ptr<tatami::NumericMatrix> dense_row, dense_column, sparse_row, sparse_column;
};

TEST_P(ModelGeneVariancesTest, UnblockedStats) {
    scran::ModelGeneVariances varfun;
    auto res = varfun.run(dense_row.get());

    auto nthreads = GetParam();
    varfun.set_num_threads(nthreads);

    if (nthreads == 1) {
        // Cursory checks.
        auto means = tatami::row_sums(dense_row.get());
        for (auto& x : means) {
            x /= dense_row->ncol();
        }
        compare_almost_equal(res.means, means);
        compare_almost_equal(res.variances, tatami::row_variances(dense_row.get()));

        for (auto f : res.fitted) {
            EXPECT_TRUE(f > 0);
        }

        int nonzero = 0; 
        for (auto f : res.residuals) {
            nonzero += (f != 0);
        }
        EXPECT_TRUE(nonzero > 0); // there is at least one non-zero residual; but we can't expect this of everyone.

    } else {
        // Checking against the same call, but parallelized.
        auto res1 = varfun.run(dense_row.get());
        EXPECT_EQ(res.means, res1.means);
        EXPECT_EQ(res.variances, res1.variances);
    }

    // Almost equal, as there are minor differences due to numerical imprecision.
    auto res2 = varfun.run(dense_column.get());
    compare_almost_equal(res.means, res2.means);
    compare_almost_equal(res.variances, res2.variances);

    auto res3 = varfun.run(sparse_row.get());
    compare_almost_equal(res.means, res3.means);
    compare_almost_equal(res.variances, res3.variances);

    auto res4 = varfun.run(sparse_column.get());
    compare_almost_equal(res.means, res4.means);
    compare_almost_equal(res.variances, res4.variances);
}

TEST_P(ModelGeneVariancesTest, BlockedStats) {
    std::vector<int> blocks(dense_row->ncol());
    for (size_t i = 0; i < blocks.size(); ++i) {
        blocks[i] = i % 3;
    }

    scran::ModelGeneVariances varfun;
    varfun.set_compute_average(false);
    auto res = varfun.run_blocked(dense_row.get(), blocks.data());
    EXPECT_TRUE(res.average.means.empty());
    EXPECT_TRUE(res.average.variances.empty());

    auto nthreads = GetParam();
    varfun.set_num_threads(nthreads);

    if (nthreads == 1) {
        // Cursory checks.
        EXPECT_EQ(res.per_block.size(), 3);
    } else {
        // Checking against the same call, but parallelized.
        auto res1 = varfun.run_blocked(dense_row.get(), blocks.data());
        for (size_t i = 0; i < 3; ++i) {
            EXPECT_EQ(res.per_block[i].means, res1.per_block[i].means);
            EXPECT_EQ(res.per_block[i].variances, res1.per_block[i].variances);
            EXPECT_EQ(res.per_block[i].fitted, res1.per_block[i].fitted);
            EXPECT_EQ(res.per_block[i].residuals, res1.per_block[i].residuals);
        }
    }

    auto res2 = varfun.run_blocked(dense_column.get(), blocks.data());
    for (size_t i = 0; i < 3; ++i) {
        compare_almost_equal(res.per_block[i].means, res2.per_block[i].means);
        compare_almost_equal(res.per_block[i].variances, res2.per_block[i].variances);
    }

    auto res3 = varfun.run_blocked(sparse_row.get(), blocks.data());
    for (size_t i = 0; i < 3; ++i) {
        compare_almost_equal(res.per_block[i].means, res3.per_block[i].means);
        compare_almost_equal(res.per_block[i].variances, res3.per_block[i].variances);
    }

    auto res4 = varfun.run_blocked(sparse_column.get(), blocks.data());
    for (size_t i = 0; i < 3; ++i) {
        compare_almost_equal(res.per_block[i].means, res3.per_block[i].means);
        compare_almost_equal(res.per_block[i].variances, res3.per_block[i].variances);
    }

    // Checking averages with equiweighting.
    varfun.set_compute_average(true);
    varfun.set_block_weight_policy(scran::WeightPolicy::EQUAL);
    {
        auto ares = varfun.run_blocked(dense_row.get(), blocks.data());
        EXPECT_EQ(ares.per_block[0].means, res.per_block[0].means);
        EXPECT_EQ(ares.per_block[1].variances, res.per_block[1].variances);
        EXPECT_EQ(ares.per_block[2].fitted, res.per_block[2].fitted);
        EXPECT_EQ(ares.per_block[1].residuals, res.per_block[1].residuals);

        std::vector<double> expected_means(dense_row->nrow()), 
            expected_variances(dense_row->nrow()),
            expected_fitted(dense_row->nrow()),
            expected_residuals(dense_row->nrow());

        for (size_t r = 0, rend = expected_means.size(); r < rend; ++r) {
            for (int b = 0; b < 3; ++b) {
                expected_means[r] += ares.per_block[b].means[r];
                expected_variances[r] += ares.per_block[b].variances[r];
                expected_fitted[r] += ares.per_block[b].fitted[r];
                expected_residuals[r] += ares.per_block[b].residuals[r];
            }

            expected_means[r] /= 3;
            expected_variances[r] /= 3;
            expected_fitted[r] /= 3;
            expected_residuals[r] /= 3;
        }

        EXPECT_EQ(expected_means, ares.average.means);
        EXPECT_EQ(expected_variances, ares.average.variances);
        EXPECT_EQ(expected_fitted, ares.average.fitted);
        EXPECT_EQ(expected_residuals, ares.average.residuals);

        // Checking limit of the variable policy.
        varfun.set_block_weight_policy(scran::WeightPolicy::VARIABLE);
        varfun.set_variable_block_weight_parameters(scran::VariableBlockWeightParameters(0, 0));

        auto vres = varfun.run_blocked(dense_row.get(), blocks.data());
        compare_almost_equal(ares.average.means, vres.average.means);
        compare_almost_equal(ares.average.variances, vres.average.variances);
        compare_almost_equal(ares.average.fitted, vres.average.fitted);
        compare_almost_equal(ares.average.residuals, vres.average.residuals);
    }

    // Checking averages without equiweighting.
    varfun.set_block_weight_policy(scran::WeightPolicy::NONE);
    {
        auto ares = varfun.run_blocked(dense_row.get(), blocks.data());
        auto block_size = scran::tabulate_ids(blocks.size(), blocks.data());

        std::vector<double> expected_means(dense_row->nrow()), 
            expected_variances(dense_row->nrow()),
            expected_fitted(dense_row->nrow()),
            expected_residuals(dense_row->nrow());

        for (size_t r = 0, rend = expected_means.size(); r < rend; ++r) {
            for (int b = 0; b < 3; ++b) {
                expected_means[r] += ares.per_block[b].means[r] * block_size[b];
                expected_variances[r] += ares.per_block[b].variances[r] * block_size[b];
                expected_fitted[r] += ares.per_block[b].fitted[r] * block_size[b];
                expected_residuals[r] += ares.per_block[b].residuals[r] * block_size[b];
            }

            expected_means[r] /= blocks.size();
            expected_variances[r] /= blocks.size();
            expected_fitted[r] /= blocks.size();
            expected_residuals[r] /= blocks.size();
        }

        compare_almost_equal(expected_means, ares.average.means);
        compare_almost_equal(expected_variances, ares.average.variances);
        compare_almost_equal(expected_fitted, ares.average.fitted);
        compare_almost_equal(expected_residuals, ares.average.residuals);

        // Checking limit of the variable policy.
        varfun.set_block_weight_policy(scran::WeightPolicy::VARIABLE);
        varfun.set_variable_block_weight_parameters(scran::VariableBlockWeightParameters(0, 100000));

        auto vres = varfun.run_blocked(dense_row.get(), blocks.data());
        compare_almost_equal(ares.average.means, vres.average.means);
        compare_almost_equal(ares.average.variances, vres.average.variances);
        compare_almost_equal(ares.average.fitted, vres.average.fitted);
        compare_almost_equal(ares.average.residuals, vres.average.residuals);
    }
}

INSTANTIATE_TEST_SUITE_P(
    ModelGeneVariances,
    ModelGeneVariancesTest,
    ::testing::Values(1, 3) // number of threads
);
