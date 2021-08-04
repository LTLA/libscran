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
    EXPECT_EQ(res.means.size(), dense_row->nrow());
    EXPECT_EQ(res.variances.size(), dense_row->nrow());

    for (auto f : res.fitted) {
        EXPECT_TRUE(f > 0);
    }
    for (auto f : res.residuals) {
        EXPECT_TRUE(f != 0);
    }

    auto res2 = var2.run(dense_column.get());
    compare_almost_equal(res.means, res2.means);
    compare_almost_equal(res.variances, res2.variances);
    compare_almost_equal(res.variances, tatami::row_variances(dense_row.get()));

    auto res3 = var3.run(sparse_row.get());
    compare_almost_equal(res.means, res3.means);
    compare_almost_equal(res.variances, res3.variances);

    auto res4 = var4.run(sparse_column.get());
    compare_almost_equal(res.means, res4.means);
    compare_almost_equal(res.variances, res4.variances);
}

TEST_F(ModelGeneVarTester, BlockedStats) {
    std::vector<int> blocks(dense_row->ncol());
    for (size_t i = 0; i < blocks.size(); ++i) {
        blocks[i] = i % 3;
    }

    auto res1 = var1.run_blocked(dense_row.get(), blocks.data());
    EXPECT_EQ(res1.means.size(), dense_row->nrow() * 3);
    EXPECT_EQ(res1.variances.size(), dense_row->nrow() * 3);

    auto res2 = var2.run_blocked(dense_column.get(), blocks.data());
    compare_almost_equal(res1.means, res2.means);
    compare_almost_equal(res1.variances, res2.variances);

    auto res3 = var3.run_blocked(sparse_row.get(), blocks.data());
    compare_almost_equal(res1.means, res3.means);
    compare_almost_equal(res1.variances, res3.variances);

    auto res4 = var4.run_blocked(sparse_column.get(), blocks.data());
    compare_almost_equal(res1.means, res4.means);
    compare_almost_equal(res1.variances, res4.variances);
}
