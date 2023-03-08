#include <gtest/gtest.h>
#include "../utils/macros.h"

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

TEST(MultiBatchMatrixTest, EigenDense) {
    size_t NR = 30, NC = 10;

    Eigen::MatrixXd thing(NR, NC);  
    std::mt19937_64 rng;
    std::normal_distribution<> dist;
    for (size_t i = 0; i < NR; ++i) {
        for (size_t j = 0; j < NC; ++j) {
            thing(i, j) = dist(rng);
        }
    }

    Eigen::VectorXd means(NC);
    for (size_t j = 0; j < NC; ++j) {
        means[j] = dist(rng);
    }

    Eigen::VectorXd weights(NR);
    for (size_t j = 0; j < NR; ++j) {
        weights[j] = std::abs(dist(rng)) + 0.1;
    }

    scran::MultiBatchEigenMatrix<decltype(thing)> batched(&thing, &weights, &means);
    auto realized = batched.realize();

    // Trying in the normal orientation.
    {
        Eigen::VectorXd rhs(NC);
        for (size_t i = 0; i < NC; ++i) {
            rhs[i] = dist(rng);
        }

        Eigen::VectorXd prod1(NR);
        batched.multiply(rhs, prod1);
        Eigen::VectorXd prod2 = realized * rhs;
        compare_almost_equal(prod1, prod2);
    }

    // Trying in the transposed orientation.
    {
        Eigen::VectorXd rhs(NR);
        for (size_t i = 0; i < NR; ++i) {
            rhs[i] = dist(rng);
        }

        Eigen::VectorXd tprod1(NC);
        batched.adjoint_multiply(rhs, tprod1);
        Eigen::VectorXd tprod2 = realized.adjoint() * rhs;
        compare_almost_equal(tprod1, tprod2);
    }
}

TEST(MultiBatchMatrixTest, CustomSparse) {
    size_t NR = 30, NC = 10;

    scran::pca_utils::CustomSparseMatrix thing(NR, NC, 1);  
    std::vector<double> values;
    std::vector<int> indices;
    std::vector<size_t> ptrs(NC + 1);

    std::mt19937_64 rng;
    std::normal_distribution<> ndist;
    std::uniform_real_distribution<> udist(0,1);
    Eigen::MatrixXd ref(NR, NC);  
    ref.setZero();

    for (size_t c = 0; c < NC; ++c) {
        for (size_t r = 0; r < NR; ++r) {
            if (udist(rng) < 0.2) {
                auto val = ndist(rng);
                ref(r, c) = val;
                values.push_back(val);
                indices.push_back(r);
                ++ptrs[c+1];
            }
        }
    }

    for (size_t i = 0; i < NC; ++i) {
        ptrs[i+1] += ptrs[i];
    }

    Eigen::VectorXd means(NC);
    for (size_t j = 0; j < NC; ++j) {
        means[j] = ndist(rng);
    }

    Eigen::VectorXd weights(NR);
    for (size_t j = 0; j < NR; ++j) {
        weights[j] = std::abs(ndist(rng)) + 0.1;
    }

    thing.fill_direct(std::move(values), std::move(indices), std::move(ptrs));
    scran::MultiBatchEigenMatrix<decltype(thing)> batched(&thing, &weights, &means);
    auto realized = batched.realize();

    // Checking that the reference matches up.
    {
        scran::MultiBatchEigenMatrix<decltype(ref)> batchedref(&ref, &weights, &means);
        auto realizedref = batchedref.realize();

        for (Eigen::Index i = 0; i < realizedref.cols(); ++i) {
            Eigen::VectorXd refcol = realizedref.col(i);
            Eigen::VectorXd obscol = realized.col(i);
            expect_equal_vectors(refcol, obscol);
        }
    }

    // Trying in the normal orientation.
    {
        Eigen::VectorXd rhs(NC);
        for (size_t i = 0; i < NC; ++i) {
            rhs[i] = ndist(rng);
        }

        Eigen::VectorXd prod1(NR);
        batched.multiply(rhs, prod1);
        Eigen::VectorXd prod2 = realized * rhs;
        compare_almost_equal(prod1, prod2);
    }

    // Trying in the transposed orientation.
    {
        Eigen::VectorXd rhs(NR);
        for (size_t i = 0; i < NR; ++i) {
            rhs[i] = ndist(rng);
        }

        Eigen::VectorXd tprod1(NC);
        batched.adjoint_multiply(rhs, tprod1);
        Eigen::VectorXd tprod2 = realized.adjoint() * rhs;
        compare_almost_equal(tprod1, tprod2);
    }
}

/******************************************/

class MultiBatchPCATestCore {
protected:
    std::shared_ptr<tatami::NumericMatrix> dense_row, dense_column, sparse_row, sparse_column;

    template<class Param>
    void assemble(const Param& param) {
        scale = std::get<0>(param);
        rank = std::get<1>(param);
        nblocks = std::get<2>(param);

        dense_row = std::unique_ptr<tatami::NumericMatrix>(new tatami::DenseRowMatrix<double>(sparse_nrow, sparse_ncol, sparse_matrix));
        dense_column = tatami::convert_to_dense(dense_row.get(), 1);
        sparse_row = tatami::convert_to_sparse(dense_row.get(), 0);
        sparse_column = tatami::convert_to_sparse(dense_row.get(), 1);
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

/******************************************/

class MultiBatchPCABasicTest : public ::testing::TestWithParam<std::tuple<bool, int, int, int, bool> >, public MultiBatchPCATestCore {};

TEST_P(MultiBatchPCABasicTest, Basic) {
    auto param = GetParam();
    assemble(param);
    int nthreads = std::get<3>(param);
    auto use_eigen = std::get<4>(param);

    scran::MultiBatchPCA runner;
    runner.set_scale(scale).set_rank(rank);
    auto block = generate_blocks(dense_row->ncol(), nblocks);
    auto ref = runner.run(dense_row.get(), block.data());

    runner.set_num_threads(nthreads);
    runner.set_use_eigen(use_eigen);

    if (use_eigen) {
        // Results should be similar with Eigen matrices.
        auto res1 = runner.run(dense_row.get(), block.data());
        expect_equal_pcs(ref.pcs, res1.pcs);
        expect_equal_vectors(ref.variance_explained, res1.variance_explained);
        EXPECT_FLOAT_EQ(ref.total_variance, res1.total_variance);

    } else if (nthreads == 1) {
        EXPECT_EQ(ref.variance_explained.size(), rank);
        EXPECT_EQ(ref.pcs.rows(), rank);
        EXPECT_EQ(ref.pcs.cols(), dense_row->ncol());

        auto drealized = runner.test_realize(dense_row.get(), block.data());
        EXPECT_EQ(drealized.rows(), dense_row->ncol()); // transposed.
        EXPECT_EQ(drealized.cols(), dense_row->nrow());

        auto srealized = runner.test_realize(sparse_column.get(), block.data()); // Check for consistency with sparse realization.
        EXPECT_EQ(srealized.rows(), dense_row->ncol()); 
        EXPECT_EQ(srealized.cols(), dense_row->nrow());

        double total_var = 0;
        for (size_t c = 0; c < drealized.cols(); ++c) {
            auto dcol = drealized.col(c);
            auto dss = dcol.squaredNorm();
            total_var += dss;

            auto scol = srealized.col(c);
            EXPECT_FLOAT_EQ(dss, scol.squaredNorm()); 

            // Note that this matrix isn't centered if batches aren't balanced; 
            // this is a bit unusual for PCA, but it's fine, as we think about 
            // weights with respect to the gene-gene covariance matrix anyway,
            // and the SVD is just a treated as an eigendecomposition of that.
            if (dense_row->ncol() % nblocks == 0) {
                EXPECT_TRUE(std::abs(dcol.sum()) < 0.000000001);
                EXPECT_TRUE(std::abs(scol.sum()) < 0.000000001);
            }
        }

        EXPECT_FLOAT_EQ(ref.total_variance, total_var); // checking the variance calculations are consistent.
        if (scale) {
            EXPECT_FLOAT_EQ(ref.total_variance, dense_row->nrow());
        }

    } else {
        // Results should be EXACTLY the same with parallelization.
        auto res1 = runner.run(dense_row.get(), block.data());
        EXPECT_EQ(ref.pcs, res1.pcs);
        EXPECT_EQ(ref.variance_explained, res1.variance_explained);
        EXPECT_EQ(ref.total_variance, res1.total_variance);
    }

    auto res2 = runner.run(dense_column.get(), block.data());
    expect_equal_pcs(ref.pcs, res2.pcs);
    expect_equal_vectors(ref.variance_explained, res2.variance_explained);
    EXPECT_FLOAT_EQ(ref.total_variance, res2.total_variance);

    auto res3 = runner.run(sparse_row.get(), block.data());
    expect_equal_pcs(ref.pcs, res3.pcs);
    expect_equal_vectors(ref.variance_explained, res3.variance_explained);
    EXPECT_FLOAT_EQ(ref.total_variance, res3.total_variance);

    auto res4 = runner.run(sparse_column.get(), block.data());
    expect_equal_pcs(ref.pcs, res4.pcs);
    expect_equal_vectors(ref.variance_explained, res4.variance_explained);
    EXPECT_FLOAT_EQ(ref.total_variance, res4.total_variance);
}

INSTANTIATE_TEST_SUITE_P(
    MultiBatchPCA,
    MultiBatchPCABasicTest,
    ::testing::Combine(
        ::testing::Values(false, true), // to scale or not to scale?
        ::testing::Values(2, 3, 4), // number of PCs to obtain
        ::testing::Values(1, 2, 3), // number of blocks
        ::testing::Values(1, 3), // number of threads
        ::testing::Values(false, true) // use Eigen for testing?
    )
);

/******************************************/

class MultiBatchPCAMoreTest : public ::testing::TestWithParam<std::tuple<bool, int, int> >, public MultiBatchPCATestCore {};

TEST_P(MultiBatchPCAMoreTest, BalancedBlock) {
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

    // Only relative variances make sense with the various scaling effects.
    for (auto& p : res1.variance_explained) { p /= res1.total_variance; }
    for (auto& p : res2.variance_explained) { p /= res2.total_variance; }

    // Need to adjust for global differences in scale; using the Frobenius norm
    // (always positive) to safely compute the scaling factor.
    if (scale) { 
        double mult = res1.pcs.norm() / res2.pcs.norm();
        res1.pcs /= mult;
    }

    if (dense_row->ncol() % nblocks == 0) { // balanced blocks.
        expect_equal_vectors(res1.variance_explained, res2.variance_explained);
        expect_equal_pcs(res1.pcs, res2.pcs);
    } else { // check that blocking actually has an effect.
        EXPECT_TRUE(std::abs(res1.pcs(0,0) - res2.pcs(0,0)) > 1e-8);
        EXPECT_TRUE(std::abs(res1.variance_explained[0] - res2.variance_explained[0]) > 1e-8);
    }
}

TEST_P(MultiBatchPCAMoreTest, DuplicatedBlocks) {
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

    EXPECT_EQ(res2.pcs.cols(), res1.pcs.cols() + subset.size()); 
    expect_equal_pcs(res1.pcs, res2.pcs.leftCols(res1.pcs.cols()));
    expect_equal_vectors(res1.variance_explained, res2.variance_explained);
    EXPECT_FLOAT_EQ(res1.total_variance, res2.total_variance);
}

TEST_P(MultiBatchPCAMoreTest, SubsetTest) {
    assemble(GetParam());

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

    scran::MultiBatchPCA runner;
    runner.set_scale(scale).set_rank(rank);

    auto block = generate_blocks(dense_row->ncol(), 3);
    auto out = runner.run(dense_row.get(), block.data(), subset.data());
    EXPECT_EQ(out.variance_explained.size(), rank);

    // Manually subsetting.
    auto mat = std::shared_ptr<tatami::NumericMatrix>(new tatami::DenseRowMatrix<double, int>(sub_nrows, dense_row->ncol(), std::move(submatrix)));
    auto ref = runner.run(mat.get(), block.data());

    expect_equal_pcs(ref.pcs, out.pcs);
    expect_equal_vectors(ref.variance_explained, out.variance_explained);
    EXPECT_FLOAT_EQ(out.total_variance, ref.total_variance);
}

INSTANTIATE_TEST_SUITE_P(
    MultiBatchPCA,
    MultiBatchPCAMoreTest,
    ::testing::Combine(
        ::testing::Values(false, true), // to scale or not to scale?
        ::testing::Values(2, 3, 4), // number of PCs to obtain
        ::testing::Values(1, 2, 3) // number of blocks
    )
);
