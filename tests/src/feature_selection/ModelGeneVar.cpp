#include <gtest/gtest.h>

#include "../data/data.h"

#include "tatami/base/DenseMatrix.hpp"
#include "tatami/base/DelayedSubset.hpp"
#include "tatami/utils/convert_to_dense.hpp"
#include "tatami/utils/convert_to_sparse.hpp"
#include "tatami/stats/sums.hpp"

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

TEST_F(ModelGeneVarTester, Unblocked) {
    auto res = var1.run(dense_row.get());
    auto res2 = var2.run(dense_column.get());
    almost_equal(res.means[0], res2.means[0]);
    almost_equal(res.variances[0], res2.variances[0]);

    auto res3 = var3.run(sparse_row.get());
    almost_equal(res.means[0], res3.means[0]);
    almost_equal(res.variances[0], res3.variances[0]);

    auto res4 = var4.run(sparse_column.get());
    almost_equal(res.means[0], res4.means[0]);
    almost_equal(res.variances[0], res4.variances[0]);
}
