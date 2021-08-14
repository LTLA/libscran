#include <gtest/gtest.h>

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

class ModelGeneVarTester : public ::testing::Test {
protected:
    void SetUp() {
        dense_row = std::unique_ptr<tatami::NumericMatrix>(new tatami::DenseRowMatrix<double>(sparse_nrow, sparse_ncol, sparse_matrix));
        dense_column = tatami::convert_to_dense(dense_row.get(), false);
        sparse_row = tatami::convert_to_sparse(dense_row.get(), true);
        sparse_column = tatami::convert_to_sparse(dense_row.get(), false);
    }
protected:
    std::shared_ptr<tatami::NumericMatrix> dense_row, dense_column, sparse_row, sparse_column;
    scran::ModelGeneVar var1, var2, var3, var4;
};

TEST_F(ModelGeneVarTester, UnblockedStats) {
    auto res = var1.run(dense_row.get());
    EXPECT_EQ(res.means.size(), 1);
    EXPECT_EQ(res.variances.size(), 1);
    EXPECT_EQ(res.means[0].size(), dense_row->nrow());
    EXPECT_EQ(res.variances[0].size(), dense_row->nrow());

    for (auto f : res.fitted[0]) {
        EXPECT_TRUE(f > 0);
    }
    for (auto f : res.residuals[0]) {
        EXPECT_TRUE(f != 0);
    }

    auto res2 = var2.run(dense_column.get());
    compare_almost_equal(res.means[0], res2.means[0]);
    compare_almost_equal(res.variances[0], res2.variances[0]);
    compare_almost_equal(res.variances[0], tatami::row_variances(dense_row.get()));

    auto res3 = var3.run(sparse_row.get());
    compare_almost_equal(res.means[0], res3.means[0]);
    compare_almost_equal(res.variances[0], res3.variances[0]);

    auto res4 = var4.run(sparse_column.get());
    compare_almost_equal(res.means[0], res4.means[0]);
    compare_almost_equal(res.variances[0], res4.variances[0]);
}

TEST_F(ModelGeneVarTester, BlockedStats) {
    std::vector<int> blocks(dense_row->ncol());
    for (size_t i = 0; i < blocks.size(); ++i) {
        blocks[i] = i % 3;
    }

    auto res1 = var1.run_blocked(dense_row.get(), blocks.data());
    EXPECT_EQ(res1.means.size(), 3);
    EXPECT_EQ(res1.variances.size(), 3);

    auto res2 = var2.run_blocked(dense_column.get(), blocks.data());
    for (size_t i = 0; i < 3; ++i) {
        compare_almost_equal(res1.means[i], res2.means[i]);
        compare_almost_equal(res1.variances[i], res2.variances[i]);
    }

    auto res3 = var3.run_blocked(sparse_row.get(), blocks.data());
    for (size_t i = 0; i < 3; ++i) {
        compare_almost_equal(res1.means[i], res3.means[i]);
        compare_almost_equal(res1.variances[i], res3.variances[i]);
    }

    auto res4 = var4.run_blocked(sparse_column.get(), blocks.data());
    for (size_t i = 0; i < 3; ++i) {
        compare_almost_equal(res1.means[i], res4.means[i]);
        compare_almost_equal(res1.variances[i], res4.variances[i]);
    }
}
