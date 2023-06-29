#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "compare_pcs.h"
#include "../data/Simulator.hpp"

#include "tatami/tatami.hpp"

#include "scran/dimensionality_reduction/MultiBatchPCA.hpp"
#include "scran/dimensionality_reduction/SimplePca.hpp"

class MultiBatchPCATestCore {
protected:
    std::shared_ptr<tatami::NumericMatrix> dense_row;

    template<class Param>
    void assemble(const Param& param) {
        scale = std::get<0>(param);
        rank = std::get<1>(param);
        nblocks = std::get<2>(param);

        size_t nr = 151, nc = 95;
        Simulator sim;
        sim.seed = nr * nc + scale + rank * nblocks;

        auto mat = sim.matrix(nr, nc);
        dense_row.reset(new decltype(mat)(std::move(mat)));
        return;
    }

    bool scale;
    int rank;
    int nblocks;
};

/******************************************/

class MultiBatchPCABasicTest : public ::testing::TestWithParam<std::tuple<bool, int, int, int> >, public MultiBatchPCATestCore {
protected:
    std::shared_ptr<tatami::NumericMatrix> dense_column, sparse_row, sparse_column;

    template<class Param>
    void extra_assemble(const Param& param) {
        assemble(param);
        dense_column = tatami::convert_to_dense(dense_row.get(), 1);
        sparse_row = tatami::convert_to_sparse(dense_row.get(), 0);
        sparse_column = tatami::convert_to_sparse(dense_row.get(), 1);
    }
};

TEST_P(MultiBatchPCABasicTest, WeightedOnly) {
    auto param = GetParam();
    extra_assemble(param);
    int nthreads = std::get<3>(param);

    scran::MultiBatchPCA runner;
    runner.set_scale(scale).set_rank(rank);
    auto block = generate_blocks(dense_row->ncol(), nblocks);
    auto ref = runner.run(dense_row.get(), block.data());

    if (nthreads == 1) {
        EXPECT_EQ(ref.variance_explained.size(), rank);
        EXPECT_EQ(ref.pcs.rows(), rank);
        EXPECT_EQ(ref.pcs.cols(), dense_row->ncol());

        are_pcs_centered(ref.pcs);
        EXPECT_TRUE(ref.total_variance >= std::accumulate(ref.variance_explained.begin(), ref.variance_explained.end(), 0.0));

        // Total variance makes sense. 
        if (scale) {
            EXPECT_FLOAT_EQ(dense_row->nrow(), ref.total_variance);
        } else {
            auto collected = fragment_matrices_by_block(dense_row, block, nblocks);

            std::vector<double> grand_mean(dense_row->nrow());
            for (int b = 0; b < collected.size(); ++b) {
                const auto& sub = collected[b];
                auto sums = tatami::row_sums(sub.get());
                for (size_t r = 0; r < grand_mean.size(); ++r) {
                    grand_mean[r] += sums[r] / sub->ncol();
                }
            }
            for (auto& x : grand_mean) {
                x /= collected.size();
            }

            double total_var = 0;
            for (int b = 0; b < collected.size(); ++b) {
                const auto& sub = collected[b];
                auto ext = sub->dense_row();
                size_t NR = sub->nrow(), NC = sub->ncol();

                double subvar = 0;
                for (size_t r = 0; r < NR; ++r) {
                    auto fetched = ext->fetch(r);
                    for (size_t c = 0; c < NC; ++c) {
                        double diff = fetched[c] - grand_mean[r];
                        subvar += diff * diff; 
                    }
                }

                total_var += subvar / NC;
            }

            EXPECT_FLOAT_EQ(total_var, ref.total_variance);
        }

    } else {
        runner.set_num_threads(nthreads);

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
        ::testing::Values(1, 3) // number of threads
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
    scran::SimplePca ref;
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

    // Mocking up the expected results.
    Eigen::MatrixXd expanded_pcs(rank, com->ncol());
    size_t offset = dense_row->ncol();
    expanded_pcs.leftCols(offset) = res1.pcs;
    for (auto x : subset) {
        expanded_pcs.col(offset) = res1.pcs.col(x);
        ++offset;
    }

    Eigen::VectorXd recenters = expanded_pcs.rowwise().sum();
    recenters /= expanded_pcs.cols();
    for (size_t i = 0, end = expanded_pcs.cols(); i < end; ++i) {
        expanded_pcs.col(i) -= recenters;
    }

    // Comparing:
    expect_equal_pcs(expanded_pcs, res2.pcs);
    expect_equal_vectors(res1.variance_explained, res2.variance_explained);
    EXPECT_FLOAT_EQ(res1.total_variance, res2.total_variance);
}

TEST_P(MultiBatchPCAMoreTest, SubsetTest) {
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
