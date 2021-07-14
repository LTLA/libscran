#include <gtest/gtest.h>

#include "../data/data.h"
#include "../utils/compare_almost_equal.h"

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

std::vector<double> eigen2vector(const Eigen::MatrixXd& input) {
    auto ptr = input.data();
    return std::vector<double>(ptr, ptr + input.rows() * input.cols());
}

std::vector<double> eigen2vector(const Eigen::VectorXd& input) {
    return std::vector<double>(input.begin(), input.end());
}

TEST_F(RunPCATester, Test) {
    scran::RunPCA runner;
    runner.set_rank(3);

    // Checking that we get more-or-less the same results. 
    auto res1 = runner.run(dense_row.get());
    std::vector<double> pcs1(eigen2vector(res1.pcs));
    std::vector<double> var1(eigen2vector(res1.variance_explained));
    EXPECT_EQ(var1.size(), 3);

    auto res2 = runner.run(dense_column.get());
    std::vector<double> pcs2(eigen2vector(res2.pcs));
    std::vector<double> var2(eigen2vector(res2.variance_explained));
    EXPECT_EQ(pcs1, pcs2);
    compare_almost_equal(pcs1, pcs2);
    compare_almost_equal(var1, var2);
    EXPECT_FLOAT_EQ(res1.total_variance, res2.total_variance);

    auto res3 = runner.run(sparse_row.get());
    std::vector<double> pcs3(eigen2vector(res3.pcs));
    std::vector<double> var3(eigen2vector(res3.variance_explained));
    compare_almost_equal(pcs1, pcs3);
    compare_almost_equal(var1, var3);
    EXPECT_FLOAT_EQ(res1.total_variance, res3.total_variance);

    auto res4 = runner.run(sparse_column.get());
    std::vector<double> pcs4(eigen2vector(res4.pcs));
    std::vector<double> var4(eigen2vector(res4.variance_explained));
    compare_almost_equal(pcs1, pcs4);
    compare_almost_equal(var1, var4);
    EXPECT_FLOAT_EQ(res1.total_variance, res4.total_variance);
}
