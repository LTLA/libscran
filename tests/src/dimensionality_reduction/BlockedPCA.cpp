#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "../data/data.h"
#include "compare_pcs.h"

#include "tatami/base/Matrix.hpp"
#include "tatami/base/DenseMatrix.hpp"
#include "tatami/utils/convert_to_dense.hpp"
#include "tatami/utils/convert_to_sparse.hpp"

#include "scran/dimensionality_reduction/BlockedPCA.hpp"
#include "scran/dimensionality_reduction/RunPCA.hpp"

TEST(BlockedMatrixTest, EigenDense) {
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

    scran::BlockedEigenMatrix<decltype(thing), int> blocked(&thing, block.data(), &centers);
    auto realized = blocked.realize();

    // Trying in the normal orientation.
    {
        Eigen::VectorXd rhs(NC);
        for (size_t i = 0; i < NC; ++i) {
            rhs[i] = dist(rng);
        }

        Eigen::VectorXd prod1(NR);
        auto wrk = blocked.workspace();
        blocked.multiply(rhs, wrk, prod1);

        Eigen::MatrixXd prod2 = realized * rhs;
        compare_almost_equal(prod1, prod2);
    }

    // Trying in the transposed orientation.
    {
        Eigen::VectorXd rhs(NR);
        for (size_t i = 0; i < NR; ++i) {
            rhs[i] = dist(rng);
        }

        Eigen::VectorXd tprod1(NC);
        auto wrk = blocked.adjoint_workspace();
        blocked.adjoint_multiply(rhs, wrk, tprod1);

        Eigen::MatrixXd tprod2 = realized.adjoint() * rhs;
        compare_almost_equal(tprod1, tprod2);
    }
}

TEST(BlockedMatrixTest, CustomSparse) {
    size_t NR = 30, NC = 10, NB = 3;
    auto block = generate_blocks(NR, NB);

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

    Eigen::MatrixXd centers(NB, NC);
    for (size_t i = 0; i < NB; ++i) {
        for (size_t j = 0; j < NC; ++j) {
            centers(i, j) = ndist(rng);
        }
    }

    scran::pca_utils::SparseMatrix thing(NR, NC, std::move(values), std::move(indices), std::move(ptrs), 1);
    scran::BlockedEigenMatrix<decltype(thing), int> blocked(&thing, block.data(), &centers);
    auto realized = blocked.realize();

    // Checking that the reference matches up.
    {
        scran::BlockedEigenMatrix<decltype(ref), int> blockedref(&ref, block.data(), &centers);
        auto realizedref = blockedref.realize();

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
        auto wrk = blocked.workspace();
        blocked.multiply(rhs, wrk, prod1);

        Eigen::MatrixXd prod2 = realized * rhs;
        compare_almost_equal(prod1, prod2);
    }

    // Trying in the transposed orientation.
    {
        Eigen::VectorXd rhs(NR);
        for (size_t i = 0; i < NR; ++i) {
            rhs[i] = ndist(rng);
        }

        Eigen::VectorXd tprod1(NC);
        auto wrk = blocked.adjoint_workspace();
        blocked.adjoint_multiply(rhs, wrk, tprod1);

        Eigen::MatrixXd tprod2 = realized.adjoint() * rhs;
        compare_almost_equal(tprod1, tprod2);
    }
}

/******************************************/

class BlockedPCATestCore {
protected:
    std::shared_ptr<tatami::NumericMatrix> dense_row, dense_column, sparse_row, sparse_column;

    template<class Param>
    void assemble(Param param) {
        scale = std::get<0>(param);
        rank = std::get<1>(param);
        nblocks = std::get<2>(param);

        dense_row = std::unique_ptr<tatami::NumericMatrix>(new tatami::DenseRowMatrix<double>(sparse_nrow, sparse_ncol, sparse_matrix));
        dense_column = tatami::convert_to_dense(dense_row.get(), 1);
        sparse_row = tatami::convert_to_sparse(dense_row.get(), 0);
        sparse_column = tatami::convert_to_sparse(dense_row.get(), 1);
        return;
    }

    bool scale;
    int rank;
    int nblocks;
};

/******************************************/

class BlockedPCABasicTest : public ::testing::TestWithParam<std::tuple<bool, int, int, int> >, public BlockedPCATestCore {};

TEST_P(BlockedPCABasicTest, Basic) {
    auto param = GetParam();
    assemble(param);
    int nthreads = std::get<3>(param);

    scran::BlockedPCA runner;
    runner.set_scale(scale).set_rank(rank);
    auto block = generate_blocks(dense_row->ncol(), nblocks);
    auto ref = runner.run(dense_row.get(), block.data());

    if (nthreads == 1) {
        EXPECT_EQ(ref.pcs.rows(), rank);
        EXPECT_EQ(ref.pcs.cols(), dense_row->ncol());
        EXPECT_EQ(ref.variance_explained.size(), rank);

        // Should be centered.
        size_t NC = dense_row->ncol();
        for (int r = 0; r < rank; ++r) {
            auto ptr = ref.pcs.data() + r;

            double mean = 0;
            for (size_t c = 0; c < NC; ++c, ptr += rank) {
                mean += *ptr;
            }
            mean /= NC;
            EXPECT_TRUE(std::abs(mean) < 0.00000001);
        }

        // Total variance makes sense. Remember, this doesn't consider the
        // loss of d.f. from calculation of the block means.
        if (scale) {
            EXPECT_FLOAT_EQ(dense_row->nrow(), ref.total_variance);
        } else {
            double total_var = 0;
            for (int b = 0; b < nblocks; ++b) {
                std::vector<int> keep;
                for (size_t i = 0; i < block.size(); ++i) {
                    if (block[i] == b) {
                        keep.push_back(i);
                    }
                }

                if (keep.size() > 1) {
                    auto sub = tatami::make_DelayedSubset<1>(dense_row, keep);
                    auto vars = tatami::row_variances(sub.get());
                    total_var += std::accumulate(vars.begin(), vars.end(), 0.0) * (keep.size() - 1);
                }
            }
            EXPECT_FLOAT_EQ(total_var / (NC - 1), ref.total_variance);
        }

    } else {
        runner.set_num_threads(nthreads);

        // Results should be EXACTLY the same with parallelization.
        auto res1 = runner.run(dense_row.get(), block.data());
        EXPECT_EQ(ref.pcs, res1.pcs);
        EXPECT_EQ(ref.variance_explained, res1.variance_explained);
        EXPECT_EQ(ref.total_variance, res1.total_variance);
    }

    // Checking that we get more-or-less the same results. 
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
    BlockedPCA,
    BlockedPCABasicTest,
    ::testing::Combine(
        ::testing::Values(false, true), // to scale or not to scale?
        ::testing::Values(2, 3, 4), // number of PCs to obtain
        ::testing::Values(1, 2, 3), // number of blocks
        ::testing::Values(1, 3) // number of threads
    )
);

/******************************************/

class BlockedPCAMoreTest : public ::testing::TestWithParam<std::tuple<bool, int, int> >, public BlockedPCATestCore {};

TEST_P(BlockedPCAMoreTest, SingleBlock) {
    assemble(GetParam());
    auto block = generate_blocks(dense_row->ncol(), nblocks);

    scran::BlockedPCA runner;
    runner.set_scale(scale).set_rank(rank);
    auto res1 = runner.run(dense_row.get(), block.data());

    // Checking that we get more-or-less the same results
    // from the vanilla PCA algorithm in the absence of blocks.
    scran::RunPCA ref;
    ref.set_scale(scale).set_rank(rank);
    auto res2 = ref.run(dense_row.get());

    if (nblocks == 1) {
        expect_equal_pcs(res1.pcs, res2.pcs);
        expect_equal_vectors(res1.variance_explained, res2.variance_explained);
        EXPECT_FLOAT_EQ(res1.total_variance, res2.total_variance);
    } else {
        // check that blocking actually has an effect.
        EXPECT_TRUE(std::abs(res1.pcs(0,0) - res2.pcs(0,0)) > 1e-8);
        if (!scale) {
            EXPECT_NE(res1.total_variance, res2.total_variance);
        }
    }
}

TEST_P(BlockedPCAMoreTest, SubsetTest) {
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

    scran::BlockedPCA runner;
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
    BlockedPCA,
    BlockedPCAMoreTest,
    ::testing::Combine(
        ::testing::Values(false, true), // to scale or not to scale?
        ::testing::Values(2, 3, 4), // number of PCs to obtain
        ::testing::Values(1, 2, 3) // number of blocks
    )
);
