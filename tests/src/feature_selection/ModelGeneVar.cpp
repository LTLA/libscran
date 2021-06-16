#include <gtest/gtest.h>

#include "../data/data.h"

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
        dense_row = std::unique_ptr<tatami::numeric_matrix>(new tatami::DenseRowMatrix<double>(sparse_nrow, sparse_ncol, sparse_matrix));
        dense_column = tatami::convert_to_dense(dense_row.get(), false);
        sparse_row = tatami::convert_to_sparse(dense_row.get(), true);
        sparse_column = tatami::convert_to_sparse(dense_row.get(), false);
    }
protected:
    std::shared_ptr<tatami::numeric_matrix> dense_row, dense_column, sparse_row, sparse_column;
    scran::ModelGeneVar<int> var1, var2, var3, var4;

    void almost_equal(const std::vector<double>& left, const std::vector<double>& right) {
        ASSERT_EQ(left.size(), right.size());
        for (size_t i = 0; i < left.size(); ++i) {
            EXPECT_FLOAT_EQ(left[i], right[i]);
        }
    }
};

TEST_F(ModelGeneVarTester, UnblockedStats) {
    auto res = var1.run(dense_row.get());
    auto res2 = var2.run(dense_column.get());
    almost_equal(res.means[0], res2.means[0]);
    almost_equal(res.variances[0], res2.variances[0]);
    almost_equal(res.variances[0], tatami::row_variances(dense_row.get()));

    auto res3 = var3.run(sparse_row.get());
    almost_equal(res.means[0], res3.means[0]);
    almost_equal(res.variances[0], res3.variances[0]);

    auto res4 = var4.run(sparse_column.get());
    almost_equal(res.means[0], res4.means[0]);
    almost_equal(res.variances[0], res4.variances[0]);
}

TEST_F(ModelGeneVarTester, BlockedStats) {
    std::vector<int> blocks(dense_row->ncol());
    for (size_t i = 0; i < blocks.size(); ++i) {
        blocks[i] = i % 3;
    }

    var1.set_blocks(blocks);
    auto res1 = var1.run(dense_row.get());

    var2.set_blocks(blocks);
    auto res2 = var2.run(dense_column.get());

    var3.set_blocks(blocks);
    auto res3 = var3.run(sparse_row.get());

    var4.set_blocks(blocks);
    auto res4 = var4.run(sparse_column.get());

    for (size_t i = 0; i < 3; ++i) {
        almost_equal(res1.means[i], res2.means[i]);
        almost_equal(res1.means[i], res3.means[i]);
        almost_equal(res1.means[i], res4.means[i]);

        almost_equal(res1.variances[i], res2.variances[i]);
        almost_equal(res1.variances[i], res3.variances[i]);
        almost_equal(res1.variances[i], res4.variances[i]);
    }
}
