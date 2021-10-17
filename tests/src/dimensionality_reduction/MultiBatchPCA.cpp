#include <gtest/gtest.h>

#include "../data/data.h"
#include "compare_pcs.h"

#include "tatami/base/Matrix.hpp"
#include "tatami/base/DenseMatrix.hpp"
#include "tatami/base/DelayedBind.hpp"
#include "tatami/base/DelayedSubset.hpp"
#include "tatami/utils/convert_to_dense.hpp"
#include "tatami/utils/convert_to_sparse.hpp"

#include "scran/dimensionality_reduction/MultiBatchPCA.hpp"
#include "scran/dimensionality_reduction/RunPCA.hpp"

TEST(MultiBatchMatrixTest, Test) {
    size_t NR = 30, NC = 10;

    Eigen::MatrixXd thing(NR, NC);  
    std::mt19937_64 rng;
    std::normal_distribution<> dist;
    for (size_t i = 0; i < NR; ++i) {
        for (size_t j = 0; j < NC; ++j) {
            thing(i, j) = dist(rng);
        }
    }

    Eigen::MatrixXd means(1, NC);
    for (size_t j = 0; j < NC; ++j) {
        means(0, j) = dist(rng);
    }

    std::vector<double> weights(NR);
    for (size_t j = 0; j < NR; ++j) {
        weights[j] = dist(rng);
        weights[j] *= weights[j];
    }

    scran::MultiBatchEigenMatrix<false, decltype(thing), double> batched(thing, weights.data(), means);
    auto realized = batched.realize();

    // Trying in the normal orientation.
    {
        size_t NRHS = 2;
        Eigen::MatrixXd rhs(NC, NRHS);
        for (size_t i = 0; i < NC; ++i) {
            for (size_t j = 0; j < NRHS; ++j) {
                rhs(i, j) = dist(rng);
            }
        }

        Eigen::MatrixXd prod1 = batched * rhs;
        Eigen::MatrixXd prod2 = realized * rhs;
        compare_almost_equal(prod1, prod2);
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

        Eigen::MatrixXd tprod1 = batched.adjoint() * rhs;
        Eigen::MatrixXd tprod2 = realized.adjoint() * rhs;
        compare_almost_equal(tprod1, tprod2);
    }
}

class MultiBatchPCATester : public ::testing::TestWithParam<std::tuple<bool, int, int> > {
protected:
    std::shared_ptr<tatami::NumericMatrix> dense_row, dense_column, sparse_row, sparse_column;

    void SetUp() {
        dense_row = std::unique_ptr<tatami::NumericMatrix>(new tatami::DenseRowMatrix<double>(sparse_nrow, sparse_ncol, sparse_matrix));
        dense_column = tatami::convert_to_dense(dense_row.get(), false);
        sparse_row = tatami::convert_to_sparse(dense_row.get(), true);
        sparse_column = tatami::convert_to_sparse(dense_row.get(), false);
    }

    template<class Param>
    void assemble(Param param) {
        scale = std::get<0>(param);
        rank = std::get<1>(param);
        nblocks = std::get<2>(param);
        return;
    }

    void expect_equal_pcs(const Eigen::MatrixXd& left, const Eigen::MatrixXd& right, double tol=1e-8) {
        ASSERT_EQ(left.cols(), right.cols());
        ASSERT_EQ(left.rows(), right.rows());

        for (size_t i = 0; i < left.cols(); ++i) {
            for (size_t j = 0; j < left.rows(); ++j) {
                EXPECT_TRUE(same_same(std::abs(left(j, i)), std::abs(right(j, i)), tol));
            }
        }
    }

    bool scale;
    int rank;
    int nblocks;
};

TEST_P(MultiBatchPCATester, Basic) {
    assemble(GetParam());

    scran::MultiBatchPCA runner;
    runner.set_scale(scale).set_rank(rank);
    auto block = generate_blocks(dense_row->ncol(), nblocks);

    // Checking that we get more-or-less the same results.
    auto res1 = runner.run(dense_row.get(), block.data());
    EXPECT_EQ(res1.variance_explained.size(), rank);

    auto res2 = runner.run(dense_column.get(), block.data());
    expect_equal_pcs(res1.pcs, res2.pcs);
    expect_equal_vectors(res1.variance_explained, res2.variance_explained);
    EXPECT_FLOAT_EQ(res1.total_variance, res2.total_variance);

    auto res3 = runner.run(sparse_row.get(), block.data());
    expect_equal_pcs(res1.pcs, res3.pcs);
    expect_equal_vectors(res1.variance_explained, res3.variance_explained);
    EXPECT_FLOAT_EQ(res1.total_variance, res3.total_variance);

    auto res4 = runner.run(sparse_column.get(), block.data());
    expect_equal_pcs(res1.pcs, res4.pcs);
    expect_equal_vectors(res1.variance_explained, res4.variance_explained);
    EXPECT_FLOAT_EQ(res1.total_variance, res4.total_variance);
}

TEST_P(MultiBatchPCATester, BalancedBlock) {
    assemble(GetParam());
    auto block = generate_blocks(dense_row->ncol(), nblocks);

    scran::MultiBatchPCA runner;
    runner.set_scale(scale).set_rank(rank);
    auto res1 = runner.run(dense_row.get(), block.data());

    // Checking that we get more-or-less the same results
    // from the vanilla PCA algorithm in the absence of blocks.
    scran::RunPCA ref;
    ref.set_scale(scale).set_rank(rank);
    auto res2 = ref.run(dense_row.get());

    // Only relative values make sense with the various scaling effects.
    for (auto& p : res1.variance_explained) { p /= res1.total_variance; }
    for (auto& p : res2.variance_explained) { p /= res2.total_variance; }

    if (dense_row->ncol() % nblocks == 0) { // balanced blocks.
        if (scale) { // need to adjust for global differences in scale. 
            double mult = res1.pcs(0,0) / res2.pcs(0,0);
            res1.pcs /= mult;
        }
        expect_equal_pcs(res1.pcs, res2.pcs);
        expect_equal_vectors(res1.variance_explained, res2.variance_explained);
    } else {
        // check that blocking actually has an effect.
        EXPECT_TRUE(std::abs(res1.pcs(0,0) - res2.pcs(0,0)) > 1e-8);
        EXPECT_TRUE(std::abs(res1.variance_explained[0] - res2.variance_explained[0]) > 1e-8);
    }
}

TEST_P(MultiBatchPCATester, DuplicatedBlocks) {
    assemble(GetParam());
    auto block = generate_blocks(dense_row->ncol(), nblocks);

    scran::MultiBatchPCA runner;
    runner.set_scale(scale).set_rank(rank);
    auto res1 = runner.run(dense_row.get(), block.data());

    // Clone the first block.
    auto block2 = block;
    std::vector<int> subset;
    for (size_t b = 0; b < block.size(); ++b) {
        if (block[b] == 0) {
            subset.push_back(b);
            block2.push_back(0);
        }
    }
    EXPECT_TRUE(subset.size() > 0);

    auto subs = tatami::make_DelayedSubset<1>(dense_row, subset);
    auto com = tatami::make_DelayedBind<1>(std::vector<decltype(subs)>{ dense_row, subs });
    auto res2 = runner.run(com.get(), block2.data());

    EXPECT_EQ(res2.pcs.rows(), res1.pcs.rows() + subset.size()); 
    expect_equal_pcs(res1.pcs, res2.pcs.topRows(res1.pcs.rows()));
    expect_equal_vectors(res1.variance_explained, res2.variance_explained);
    EXPECT_FLOAT_EQ(res1.total_variance, res2.total_variance);
}

INSTANTIATE_TEST_SUITE_P(
    MultiBatchPCA,
    MultiBatchPCATester,
    ::testing::Combine(
        ::testing::Values(false, true), // to scale or not to scale?
        ::testing::Values(2, 3, 4), // number of PCs to obtain
        ::testing::Values(1, 2, 3) // number of blocks
    )
);
