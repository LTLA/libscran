#include <gtest/gtest.h>

#include "../data/data.h"
#include "../utils/compare_almost_equal.h"
#include "../utils/eigen2vector.h"

#include "tatami/base/Matrix.hpp"
#include "tatami/base/DenseMatrix.hpp"
#include "tatami/utils/convert_to_dense.hpp"
#include "tatami/utils/convert_to_sparse.hpp"

#include "scran/dimensionality_reduction/BlockedPCA.hpp"
#include "scran/dimensionality_reduction/RunPCA.hpp"

std::vector<int> generate_blocks(int nobs, int nblocks) {
    std::vector<int> blocks(nobs);
    for (int i = 0; i < nobs; ++i) {
        blocks[i] = i % nblocks;
    }
    return blocks;
}

TEST(BlockedMatrixTest, Test) {
    size_t NR = 30, NC = 10, NB = 3;
    auto block = generate_blocks(NR, NB);

    Eigen::MatrixXd thing(NR, NC);  
    std::mt19937_64 rng;
    std::normal_distribution<> dist;
    for (size_t i = 0; i < NR; ++i) {
        for (size_t j = 0; j < NC; ++j) {
            thing(i, j) = dist(rng);
        }
    }

    Eigen::MatrixXd centers(NB, NC);
    for (size_t i = 0; i < NB; ++i) {
        for (size_t j = 0; j < NC; ++j) {
            centers(i, j) = dist(rng);
        }
    }

    scran::BlockedEigenMatrix<false, decltype(thing), int> blocked(thing, block.data(), centers);
    auto realized = blocked.realize();

    // Trying in the normal orientation.
    {
        size_t NRHS = 2;
        Eigen::MatrixXd rhs(NC, NRHS);
        for (size_t i = 0; i < NC; ++i) {
            for (size_t j = 0; j < NRHS; ++j) {
                rhs(i, j) = dist(rng);
            }
        }

        Eigen::MatrixXd prod1 = blocked * rhs;
        Eigen::MatrixXd prod2 = realized * rhs;
        compare_almost_equal(eigen2vector(prod1), eigen2vector(prod2));
    }

    // Trying in the transposed orientation.
    {
        size_t NRHS = 2;
        Eigen::MatrixXd rhs(NR, NRHS);
        for (size_t i = 0; i < NR; ++i) {
            for (size_t j = 0; j < NRHS; ++j) {
                rhs(i, j) = dist(rng);
            }
        }

        Eigen::MatrixXd tprod1 = blocked.adjoint() * rhs;
        Eigen::MatrixXd tprod2 = realized.adjoint() * rhs;
        compare_almost_equal(eigen2vector(tprod1), eigen2vector(tprod2));
    }
}

class BlockedPCATester : public ::testing::Test {
protected:
    std::shared_ptr<tatami::NumericMatrix> dense_row, dense_column, sparse_row, sparse_column;

    void SetUp() {
        dense_row = std::unique_ptr<tatami::NumericMatrix>(new tatami::DenseRowMatrix<double>(sparse_nrow, sparse_ncol, sparse_matrix));
        dense_column = tatami::convert_to_dense(dense_row.get(), false);
        sparse_row = tatami::convert_to_sparse(dense_row.get(), true);
        sparse_column = tatami::convert_to_sparse(dense_row.get(), false);
    }
};

TEST_F(BlockedPCATester, Test) {
    scran::BlockedPCA runner;
    runner.set_rank(3);
    auto block = generate_blocks(dense_row->ncol(), 3);

    // Checking that we get more-or-less the same results. 
    auto res1 = runner.run(dense_row.get(), block.data());
    std::vector<double> pcs1(eigen2vector(res1.pcs));
    std::vector<double> var1(eigen2vector(res1.variance_explained));
    EXPECT_EQ(var1.size(), 3);

    auto res2 = runner.run(dense_column.get(), block.data());
    std::vector<double> pcs2(eigen2vector(res2.pcs));
    std::vector<double> var2(eigen2vector(res2.variance_explained));
    compare_almost_equal(pcs1, pcs2);
    compare_almost_equal(var1, var2);
    EXPECT_FLOAT_EQ(res1.total_variance, res2.total_variance);

    auto res3 = runner.run(sparse_row.get(), block.data());
    std::vector<double> pcs3(eigen2vector(res3.pcs));
    std::vector<double> var3(eigen2vector(res3.variance_explained));
    compare_almost_equal(pcs1, pcs3);
    compare_almost_equal(var1, var3);
    EXPECT_FLOAT_EQ(res1.total_variance, res3.total_variance);

    auto res4 = runner.run(sparse_column.get(), block.data());
    std::vector<double> pcs4(eigen2vector(res4.pcs));
    std::vector<double> var4(eigen2vector(res4.variance_explained));
    compare_almost_equal(pcs1, pcs4);
    compare_almost_equal(var1, var4);
    EXPECT_FLOAT_EQ(res1.total_variance, res4.total_variance);
}

TEST_F(BlockedPCATester, SingleBlock) {
    scran::BlockedPCA runner;
    runner.set_rank(3);
    auto block = generate_blocks(dense_row->ncol(), 1);

    auto res1 = runner.run(dense_row.get(), block.data());
    std::vector<double> pcs1(eigen2vector(res1.pcs));
    std::vector<double> var1(eigen2vector(res1.variance_explained));

    // Checking that we get more-or-less the same results
    // from the vanilla PCA algorithm in the absence of blocks.
    scran::RunPCA ref;
    ref.set_rank(3);

    auto res2 = ref.run(dense_row.get());
    std::vector<double> pcs2(eigen2vector(res2.pcs));
    std::vector<double> var2(eigen2vector(res2.variance_explained));

    compare_almost_equal(pcs1, pcs2);
    compare_almost_equal(var1, var2);
}

TEST_F(BlockedPCATester, SubsetTest) {
    std::vector<int> subset(dense_row->nrow());
    std::vector<double> buffer(dense_row->ncol());
    std::vector<double> submatrix;
    auto it = sparse_matrix.begin();

    size_t sub_nrows = 0;
    for (size_t i = 0; i < subset.size(); ++i) {
        subset[i] = i%2;
        if (subset[i]) {
            auto ptr = dense_row->row(i, buffer.data());
            submatrix.insert(submatrix.end(), ptr, ptr + dense_row->ncol());
            ++sub_nrows;
        }
    }

    scran::BlockedPCA runner;
    runner.set_rank(4);

    auto block = generate_blocks(dense_row->ncol(), 3);
    auto out = runner.run(dense_row.get(), block.data(), subset.data());
    std::vector<double> opcs(eigen2vector(out.pcs));
    std::vector<double> ovar(eigen2vector(out.variance_explained));
    EXPECT_EQ(ovar.size(), 4);

    // Manually subsetting.
    auto mat = std::shared_ptr<tatami::NumericMatrix>(new tatami::DenseRowMatrix<double, int>(sub_nrows, dense_row->ncol(), std::move(submatrix)));
    auto ref = runner.run(mat.get(), block.data());
    std::vector<double> rpcs(eigen2vector(ref.pcs));
    std::vector<double> rvar(eigen2vector(ref.variance_explained));

    compare_almost_equal(opcs, rpcs);
    compare_almost_equal(ovar, rvar);
    EXPECT_FLOAT_EQ(out.total_variance, ref.total_variance);
}
