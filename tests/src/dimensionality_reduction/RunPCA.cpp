#include <gtest/gtest.h>

#include "../data/data.h"

#include "tatami/base/Matrix.hpp"
#include "tatami/base/DenseMatrix.hpp"
#include "tatami/utils/convert_to_dense.hpp"
#include "tatami/utils/convert_to_sparse.hpp"

#include "scran/dimensionality_reduction/RunPCA.hpp"

class RunPCATester : public ::testing::Test {
protected:
    std::shared_ptr<tatami::NumericMatrix> dense_row, dense_column, sparse_row, sparse_column;

    void SetUp() {
        dense_row = std::unique_ptr<tatami::NumericMatrix>(new tatami::DenseRowMatrix<double>(sparse_nrow, sparse_ncol, sparse_matrix));
        dense_column = tatami::convert_to_dense(dense_row.get(), false);
        sparse_row = tatami::convert_to_sparse(dense_row.get(), true);
        sparse_column = tatami::convert_to_sparse(dense_row.get(), false);
    }
};

TEST_F(RunPCATester, Test) {
    scran::RunPCA runner;
    runner.set_rank(3);
    auto res1 = runner.run(dense_row.get());
    auto res2 = runner.run(dense_column.get());
    auto res3 = runner.run(sparse_row.get());
    auto res4 = runner.run(sparse_column.get());
}
