#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "../data/Simulator.hpp"
#include "compare_pcs.h"

#include "tatami/tatami.hpp"

#include "scran/dimensionality_reduction/SimplePca.hpp"

class SimplePcaTestCore {
protected:
    std::shared_ptr<tatami::NumericMatrix> dense_row, dense_column, sparse_row, sparse_column;

    template<class Param>
    void assemble(const Param& param) {
        size_t nr = 199, nc = 165;
        auto mat = Simulator().matrix(nr, nc);
        dense_row.reset(new decltype(mat)(std::move(mat)));
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

class SimplePcaBasicTest : public ::testing::TestWithParam<std::tuple<bool, int, int> >, public SimplePcaTestCore {};

TEST_P(SimplePcaBasicTest, Test) {
    auto param = GetParam();
    assemble(param);
    auto threads = std::get<2>(param);

    scran::SimplePca runner;
    runner.set_scale(scale).set_rank(rank);
    auto ref = runner.run(dense_row.get());

    if (threads == 1) {
        EXPECT_EQ(ref.variance_explained.size(), rank);
        EXPECT_EQ(ref.pcs.rows(), rank);
        EXPECT_EQ(ref.pcs.cols(), dense_row->ncol());

        // Checking that we scaled the PCs correctly.
        size_t NC = dense_row->ncol();
        for (int r = 0; r < rank; ++r) {
            auto ptr = ref.pcs.data() + r;

            double mean = 0;
            for (size_t c = 0; c < NC; ++c, ptr += rank) {
                mean += *ptr;            
            }
            mean /= NC;
            EXPECT_TRUE(std::abs(mean) < 0.00000001);

            double var = 0;
            ptr = ref.pcs.data() + r;
            for (size_t c = 0; c < NC; ++c, ptr += rank) {
                var += (*ptr - mean) * (*ptr - mean);
            }
            var /= NC - 1;

            EXPECT_FLOAT_EQ(var, ref.variance_explained[r]);
        }

        if (scale) {
            EXPECT_FLOAT_EQ(dense_row->nrow(), ref.total_variance);
        } else {
            auto vars = tatami::row_variances(dense_row.get());
            auto total_var = std::accumulate(vars.begin(), vars.end(), 0.0);
            EXPECT_FLOAT_EQ(total_var, ref.total_variance);
        }

        EXPECT_TRUE(ref.total_variance >= std::accumulate(ref.variance_explained.begin(), ref.variance_explained.end(), 0.0));

    } else {
        runner.set_num_threads(threads);

        // Results should be EXACTLY the same with parallelization.

        auto res1 = runner.run(dense_row.get());
        EXPECT_EQ(ref.pcs, res1.pcs);
        EXPECT_EQ(ref.variance_explained, res1.variance_explained);
        EXPECT_EQ(ref.total_variance, res1.total_variance);
    }

    // Checking that we get more-or-less the same results. 
    auto res2 = runner.run(dense_column.get());
    expect_equal_pcs(ref.pcs, res2.pcs);
    expect_equal_vectors(ref.variance_explained, res2.variance_explained);
    EXPECT_FLOAT_EQ(ref.total_variance, res2.total_variance);

    auto res3 = runner.run(sparse_row.get());
    expect_equal_pcs(ref.pcs, res3.pcs);
    expect_equal_vectors(ref.variance_explained, res3.variance_explained);
    EXPECT_FLOAT_EQ(ref.total_variance, res3.total_variance);

    auto res4 = runner.run(sparse_column.get());
    expect_equal_pcs(ref.pcs, res4.pcs);
    expect_equal_vectors(ref.variance_explained, res4.variance_explained);
    EXPECT_FLOAT_EQ(ref.total_variance, res4.total_variance);
}

INSTANTIATE_TEST_SUITE_P(
    SimplePca,
    SimplePcaBasicTest,
    ::testing::Combine(
        ::testing::Values(false, true), // to scale or not to scale?
        ::testing::Values(2, 3, 4), // number of PCs to obtain
        ::testing::Values(1, 3) // number of threads
    )
);

/******************************************/

class SimplePcaMoreTest : public ::testing::TestWithParam<std::tuple<bool, int> >, public SimplePcaTestCore {};

TEST_P(SimplePcaMoreTest, Subset) {
    assemble(GetParam());

    std::vector<int> subset(dense_row->nrow());
    std::vector<double> buffer(dense_row->ncol());
    std::vector<double> submatrix;

    size_t sub_nrows = 0;
    auto ext = dense_row->dense_row();
    for (size_t i = 0; i < subset.size(); ++i) {
        subset[i] = i%2;
        if (subset[i]) {
            auto ptr = ext->fetch(i, buffer.data());
            submatrix.insert(submatrix.end(), ptr, ptr + dense_row->ncol());
            ++sub_nrows;
        }
    }

    scran::SimplePca runner;
    runner.set_scale(scale).set_rank(rank).set_return_rotation(true);

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

TEST_P(SimplePcaMoreTest, ZeroVariance) {
    auto param = GetParam();
    scale = std::get<0>(param);
    rank = std::get<1>(param);

    size_t nr = 109, nc = 153;
    auto vec = Simulator().vector(nr * nc);

    auto copy = vec;
    size_t last_row = (nr - 1) * nc;
    std::fill(copy.begin() + last_row, copy.begin() + last_row + nc, 0);
    tatami::DenseRowMatrix<double> has_zero(nr, nc, std::move(copy));

    std::vector<double> removed(vec.begin(), vec.begin() + last_row);
    tatami::DenseRowMatrix<double> leftovers(nr - 1, nc, std::move(removed));

    scran::SimplePca runner;
    runner.set_scale(scale).set_rank(rank);
    
    auto ref = runner.run(&leftovers);
    auto out = runner.run(&has_zero);

    expect_equal_pcs(ref.pcs, out.pcs, 1e-4, false); // RNG effect is slightly different when we lose a feature, hence the need for a looser tolerance.
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
    SimplePca,
    SimplePcaMoreTest,
    ::testing::Combine(
        ::testing::Values(false, true), // to scale or not to scale?
        ::testing::Values(2, 3, 4) // number of PCs to obtain
    )
);

TEST(SimplePcaTest, ReturnValues) {
    size_t nr = 99, nc = 123;
    auto mat = Simulator().matrix(nr, nc);

    scran::SimplePca runner;
    runner.set_rank(4);
    {
        auto ref = runner.run(&mat);
        EXPECT_EQ(ref.rotation.cols(), 0);
        EXPECT_EQ(ref.rotation.rows(), 0);
        EXPECT_EQ(ref.center.size(), 0);
        EXPECT_EQ(ref.scale.size(), 0);
    }

    runner.set_return_center(true).set_return_scale(true).set_return_rotation(true);
    {
        auto ref = runner.run(&mat);
        EXPECT_EQ(ref.rotation.cols(), 4);
        EXPECT_EQ(ref.rotation.rows(), nr);
        EXPECT_EQ(ref.center.size(), nr);
        EXPECT_EQ(ref.scale.size(), nr);
    }
}
