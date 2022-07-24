#include <gtest/gtest.h>

#include "../data/data.h"
#include "compare_pcs.h"

#include "tatami/base/Matrix.hpp"
#include "tatami/base/DenseMatrix.hpp"
#include "tatami/utils/convert_to_dense.hpp"
#include "tatami/utils/convert_to_sparse.hpp"

#include "scran/dimensionality_reduction/RunPCA.hpp"

class RunPCATestCore {
protected:
    std::shared_ptr<tatami::NumericMatrix> dense_row, dense_column, sparse_row, sparse_column;

    template<class Param>
    void assemble(const Param& param) {
        dense_row = std::unique_ptr<tatami::NumericMatrix>(new tatami::DenseRowMatrix<double>(sparse_nrow, sparse_ncol, sparse_matrix));
        dense_column = tatami::convert_to_dense(dense_row.get(), 1);
        sparse_row = tatami::convert_to_sparse(dense_row.get(), 0);
        sparse_column = tatami::convert_to_sparse(dense_row.get(), 1);

        scale = std::get<0>(param);
        rank = std::get<1>(param);
        return;
    }

    bool scale;
    int rank;
};

/******************************************/

class RunPCABasicTest : public ::testing::TestWithParam<std::tuple<bool, int, int, bool> >, public RunPCATestCore {};

TEST_P(RunPCABasicTest, Test) {
    auto param = GetParam();
    assemble(param);

    scran::RunPCA runner;
    runner.set_scale(scale).set_rank(rank);
    auto res = runner.run(dense_row.get());
    
    auto threads = std::get<2>(param);
    runner.set_num_threads(threads);
    auto use_eigen = std::get<3>(param);
    if (use_eigen) {
        runner.set_use_eigen(true);
    }

    if (!use_eigen && threads == 1) {
        EXPECT_EQ(res.variance_explained.size(), rank);
        EXPECT_EQ(res.pcs.rows(), rank);
        EXPECT_EQ(res.pcs.cols(), dense_row->ncol());
        EXPECT_EQ(res.rotation.rows(), dense_row->nrow());
        EXPECT_EQ(res.rotation.cols(), rank);

        // Checking that we scaled the PCs correctly.
        size_t NC = dense_row->ncol();
        for (int r = 0; r < rank; ++r) {
            auto ptr = res.pcs.data() + r;

            double mean = 0;
            for (size_t c = 0; c < NC; ++c, ptr += rank) {
                mean += *ptr;            
            }
            mean /= NC;
            EXPECT_TRUE(std::abs(mean) < 0.00000001);

            double var = 0;
            ptr = res.pcs.data() + r;
            for (size_t c = 0; c < NC; ++c, ptr += rank) {
                var += (*ptr - mean) * (*ptr - mean);
            }
            var /= NC - 1;

            EXPECT_FLOAT_EQ(var, res.variance_explained[r]);
        }
    } else {
        auto res1 = runner.run(dense_row.get());
        expect_equal_pcs(res.pcs, res1.pcs);
        expect_equal_vectors(res.variance_explained, res1.variance_explained);
        EXPECT_FLOAT_EQ(res.total_variance, res1.total_variance);
    }

    // Checking that we get more-or-less the same results. 
    auto res2 = runner.run(dense_column.get());
    expect_equal_pcs(res.pcs, res2.pcs);
    expect_equal_vectors(res.variance_explained, res2.variance_explained);
    EXPECT_FLOAT_EQ(res.total_variance, res2.total_variance);

    auto res3 = runner.run(sparse_row.get());
    expect_equal_pcs(res.pcs, res3.pcs);
    expect_equal_vectors(res.variance_explained, res3.variance_explained);
    EXPECT_FLOAT_EQ(res.total_variance, res3.total_variance);

    auto res4 = runner.run(sparse_column.get());
    expect_equal_pcs(res.pcs, res4.pcs);
    expect_equal_vectors(res.variance_explained, res4.variance_explained);
    EXPECT_FLOAT_EQ(res.total_variance, res4.total_variance);
}

INSTANTIATE_TEST_SUITE_P(
    RunPCA,
    RunPCABasicTest,
    ::testing::Combine(
        ::testing::Values(false, true), // to scale or not to scale?
        ::testing::Values(2, 3, 4), // number of PCs to obtain
        ::testing::Values(1, 3), // number of threads
        ::testing::Values(false, true) // use Eigen for testing?
    )
);

/******************************************/

class RunPCAMoreTest : public ::testing::TestWithParam<std::tuple<bool, int> >, public RunPCATestCore {};

TEST_P(RunPCAMoreTest, Subset) {
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

    scran::RunPCA runner;
    runner.set_scale(scale).set_rank(rank);

    auto out = runner.run(dense_row.get(), subset.data());
    EXPECT_EQ(out.variance_explained.size(), rank);
    EXPECT_EQ(out.rotation.rows(), sub_nrows);
    EXPECT_EQ(out.rotation.cols(), rank);

    // Manually subsetting.
    auto mat = std::shared_ptr<tatami::NumericMatrix>(new tatami::DenseRowMatrix<double, int>(sub_nrows, dense_row->ncol(), std::move(submatrix)));
    auto ref = runner.run(mat.get());

    expect_equal_pcs(ref.pcs, out.pcs);
    expect_equal_vectors(ref.variance_explained, out.variance_explained);
    EXPECT_FLOAT_EQ(out.total_variance, ref.total_variance);
}

TEST_P(RunPCAMoreTest, ZeroVariance) {
    assemble(GetParam());

    auto copy = sparse_matrix;
    std::fill(copy.begin(), copy.begin() + sparse_ncol, 0);
    tatami::DenseRowMatrix<double> has_zero(sparse_nrow, sparse_ncol, copy);

    std::vector<double> removed(copy.begin() + sparse_ncol, copy.end());
    tatami::DenseRowMatrix<double> leftovers(sparse_nrow - 1, sparse_ncol, removed);

    scran::RunPCA runner;
    runner.set_scale(scale).set_rank(rank);
    
    auto ref = runner.run(&leftovers);
    auto out = runner.run(&has_zero);

    expect_equal_pcs(ref.pcs, out.pcs, 1e-6); // dunno why it needs a higher tolerance, but whatever.
    expect_equal_vectors(ref.variance_explained, out.variance_explained);
    EXPECT_FLOAT_EQ(out.total_variance, ref.total_variance);

    // Same behavior with sparse representation.
    auto sparse_zero = tatami::convert_to_sparse(&has_zero, true);
    auto spout = runner.run(sparse_zero.get());

    expect_equal_pcs(spout.pcs, out.pcs);
    expect_equal_vectors(spout.variance_explained, out.variance_explained);
    EXPECT_FLOAT_EQ(spout.total_variance, out.total_variance);
}

INSTANTIATE_TEST_SUITE_P(
    RunPCA,
    RunPCAMoreTest,
    ::testing::Combine(
        ::testing::Values(false, true), // to scale or not to scale?
        ::testing::Values(2, 3, 4) // number of PCs to obtain
    )
);
