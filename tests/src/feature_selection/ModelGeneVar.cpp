#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "../data/data.h"
#include "../utils/compare_almost_equal.h"

#include "tatami/base/DenseMatrix.hpp"
#include "tatami/base/DelayedSubset.hpp"
#include "tatami/utils/convert_to_dense.hpp"
#include "tatami/utils/convert_to_sparse.hpp"
#include "tatami/stats/sums.hpp"
#include "tatami/stats/variances.hpp"

#include "scran/feature_selection/ModelGeneVar.hpp"

#include <cmath>

class ModelGeneVarTest : public ::testing::TestWithParam<int> {
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

TEST_P(ModelGeneVarTest, UnblockedStats) {
    scran::ModelGeneVar varfun;
    auto res = varfun.run(dense_row.get());

    auto nthreads = GetParam();
    varfun.set_num_threads(nthreads);

    if (nthreads == 1) {
        // Cursory checks.
        EXPECT_EQ(res.means.size(), 1);
        EXPECT_EQ(res.variances.size(), 1);
        EXPECT_EQ(res.means[0].size(), dense_row->nrow());
        EXPECT_EQ(res.variances[0].size(), dense_row->nrow());

        for (auto f : res.fitted[0]) {
            EXPECT_TRUE(f > 0);
        }
        for (auto f : res.residuals[0]) {
            std::cout << f<< std::endl;
            EXPECT_TRUE(f != 0);
        }

        compare_almost_equal(res.variances[0], tatami::row_variances(dense_row.get()));

    } else {
        // Checking against the same call, but parallelized.
        auto res1 = varfun.run(dense_row.get());
        EXPECT_EQ(res.means[0], res1.means[0]);
        EXPECT_EQ(res.variances[0], res1.variances[0]);
    }

    // Almost equal, as there are minor differences due to numerical imprecision.
    auto res2 = varfun.run(dense_column.get());
    compare_almost_equal(res.means[0], res2.means[0]);
    compare_almost_equal(res.variances[0], res2.variances[0]);

    auto res3 = varfun.run(sparse_row.get());
    compare_almost_equal(res.means[0], res3.means[0]);
    compare_almost_equal(res.variances[0], res3.variances[0]);

    auto res4 = varfun.run(sparse_column.get());
    compare_almost_equal(res.means[0], res4.means[0]);
    compare_almost_equal(res.variances[0], res4.variances[0]);
}

TEST_P(ModelGeneVarTest, BlockedStats) {
    std::vector<int> blocks(dense_row->ncol());
    for (size_t i = 0; i < blocks.size(); ++i) {
        blocks[i] = i % 3;
    }

    scran::ModelGeneVar varfun;
    auto res = varfun.run_blocked(dense_row.get(), blocks.data());

    auto nthreads = GetParam();
    varfun.set_num_threads(nthreads);

    if (nthreads == 1) {
        // Cursory checks.
        EXPECT_EQ(res.means.size(), 3);
        EXPECT_EQ(res.variances.size(), 3);
    } else {
        // Checking against the same call, but parallelized.
        auto res1 = varfun.run_blocked(dense_row.get(), blocks.data());
        for (size_t i = 0; i < 3; ++i) {
            EXPECT_EQ(res.means[i], res1.means[i]);
            EXPECT_EQ(res.variances[i], res1.variances[i]);
        }
    }

    auto res2 = varfun.run_blocked(dense_column.get(), blocks.data());
    for (size_t i = 0; i < 3; ++i) {
        compare_almost_equal(res.means[i], res2.means[i]);
        compare_almost_equal(res.variances[i], res2.variances[i]);
    }

    auto res3 = varfun.run_blocked(sparse_row.get(), blocks.data());
    for (size_t i = 0; i < 3; ++i) {
        compare_almost_equal(res.means[i], res3.means[i]);
        compare_almost_equal(res.variances[i], res3.variances[i]);
    }

    auto res4 = varfun.run_blocked(sparse_column.get(), blocks.data());
    for (size_t i = 0; i < 3; ++i) {
        compare_almost_equal(res.means[i], res4.means[i]);
        compare_almost_equal(res.variances[i], res4.variances[i]);
    }
}

INSTANTIATE_TEST_CASE_P(
    ModelGeneVar,
    ModelGeneVarTest,
    ::testing::Values(1, 3) // number of threads
);
