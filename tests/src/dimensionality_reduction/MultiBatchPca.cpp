#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "compare_pcs.h"
#include "../data/Simulator.hpp"

#include "tatami/tatami.hpp"

#include "scran/dimensionality_reduction/MultiBatchPca.hpp"
#include "scran/dimensionality_reduction/SimplePca.hpp"
#include "scran/dimensionality_reduction/ResidualPca.hpp"

class MultiBatchPcaTestCore {
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

class MultiBatchPcaBasicTest : public ::testing::TestWithParam<std::tuple<bool, int, int, int> >, public MultiBatchPcaTestCore {
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

TEST_P(MultiBatchPcaBasicTest, WeightedOnly) {
    auto param = GetParam();
    extra_assemble(param);
    int nthreads = std::get<3>(param);

    scran::MultiBatchPca runner;
    runner.set_scale(scale).set_rank(rank);
    runner.set_block_weight_policy(scran::WeightPolicy::EQUAL);

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

            EXPECT_FLOAT_EQ(total_var / (dense_row->ncol() - 1), ref.total_variance);
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

TEST_P(MultiBatchPcaBasicTest, ResidualOnly) {
    auto param = GetParam();
    extra_assemble(param);
    int nthreads = std::get<3>(param);

    scran::MultiBatchPca runner;
    runner.set_scale(scale).set_rank(rank);
    runner.set_use_residuals(true);
    runner.set_block_weight_policy(scran::WeightPolicy::NONE);

    auto block = generate_blocks(dense_row->ncol(), nblocks);
    auto ref = runner.run(dense_row.get(), block.data());

    if (nthreads == 1) {
        are_pcs_centered(ref.pcs);
        EXPECT_TRUE(ref.total_variance >= std::accumulate(ref.variance_explained.begin(), ref.variance_explained.end(), 0.0));

        // Total variance makes sense. 
        if (scale) {
            EXPECT_FLOAT_EQ(dense_row->nrow(), ref.total_variance);
        } else {
            auto collected = fragment_matrices_by_block(dense_row, block, nblocks);

            double total_var = 0;
            for (int b = 0, end = collected.size(); b < end; ++b) {
                const auto& sub = collected[b];
                auto vars = tatami::row_variances(sub.get());
                total_var += std::accumulate(vars.begin(), vars.end(), 0.0) * (sub->ncol() - 1);
            }

            EXPECT_FLOAT_EQ(total_var / (dense_row->ncol() - 1), ref.total_variance);
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

TEST_P(MultiBatchPcaBasicTest, WeightedResidual) {
    auto param = GetParam();
    extra_assemble(param);
    int nthreads = std::get<3>(param);

    scran::MultiBatchPca runner;
    runner.set_scale(scale).set_rank(rank);
    runner.set_use_residuals(true);
    runner.set_block_weight_policy(scran::WeightPolicy::EQUAL);

    auto block = generate_blocks(dense_row->ncol(), nblocks);
    auto ref = runner.run(dense_row.get(), block.data());

    if (nthreads == 1) {
        are_pcs_centered(ref.pcs);
        EXPECT_TRUE(ref.total_variance >= std::accumulate(ref.variance_explained.begin(), ref.variance_explained.end(), 0.0));

        // Total variance makes sense. 
        if (scale) {
            EXPECT_FLOAT_EQ(dense_row->nrow(), ref.total_variance);
        } else {
            auto collected = fragment_matrices_by_block(dense_row, block, nblocks);

            double total_var = 0;
            for (int b = 0, end = collected.size(); b < end; ++b) {
                const auto& sub = collected[b];
                auto vars = tatami::row_variances(sub.get());
                total_var += std::accumulate(vars.begin(), vars.end(), 0.0) * static_cast<double>(sub->ncol() - 1) / sub->ncol();
            }

            EXPECT_FLOAT_EQ(total_var / (dense_row->ncol() - 1), ref.total_variance);
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
    MultiBatchPca,
    MultiBatchPcaBasicTest,
    ::testing::Combine(
        ::testing::Values(false, true), // to scale or not to scale?
        ::testing::Values(2, 3, 4), // number of PCs to obtain
        ::testing::Values(1, 2, 3), // number of blocks
        ::testing::Values(1, 3) // number of threads
    )
);

/******************************************/

class MultiBatchPcaMoreTest : public ::testing::TestWithParam<std::tuple<bool, int, int> >, public MultiBatchPcaTestCore {};

TEST_P(MultiBatchPcaMoreTest, WeightedOnly_VersusSimple) {
    assemble(GetParam());

    std::vector<int> block;
    std::vector<std::shared_ptr<tatami::NumericMatrix> > combined;
    combined.reserve(nblocks);
    for (int b = 0; b < nblocks; ++b) {
        combined.push_back(dense_row);
        block.insert(block.end(), dense_row->ncol(), b);
    }
    auto expanded = tatami::make_DelayedBind<1>(std::move(combined));

    scran::MultiBatchPca runner;
    runner.set_scale(scale).set_rank(rank);
    runner.set_block_weight_policy(scran::WeightPolicy::EQUAL);
    auto res1 = runner.run(expanded.get(), block.data());

    // Checking that we get more-or-less the same results
    // from the vanilla PCA algorithm in the absence of blocks.
    scran::SimplePca ref;
    ref.set_scale(scale).set_rank(rank);
    auto res2 = ref.run(expanded.get());

    // Only relative variances make sense with the various scaling effects.
    for (auto& p : res1.variance_explained) { p /= res1.total_variance; }
    for (auto& p : res2.variance_explained) { p /= res2.total_variance; }

    // Need to adjust for global differences in scale; using the Frobenius norm
    // (always positive) to safely compute the scaling factor.
    if (scale) { 
        double mult = res1.pcs.norm() / res2.pcs.norm();
        res1.pcs /= mult;
    }

    expect_equal_vectors(res1.variance_explained, res2.variance_explained);
    expect_equal_pcs(res1.pcs, res2.pcs);
}

TEST_P(MultiBatchPcaMoreTest, WeightedOnly_DuplicatedBlocks) {
    assemble(GetParam());
    auto block = generate_blocks(dense_row->ncol(), nblocks);

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

    scran::MultiBatchPca runner;
    runner.set_scale(scale).set_rank(rank);

    // Checking what happens when every batch is equally weighted.
    runner.set_block_weight_policy(scran::WeightPolicy::EQUAL);
    {
        auto res2 = runner.run(com.get(), block2.data());
        res2.pcs.array() /= res2.pcs.norm();
        res2.variance_explained.array() /= res2.total_variance;

        {
            // Mocking up the expected results.
            auto res1 = runner.run(dense_row.get(), block.data());

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
            expanded_pcs.array() /= expanded_pcs.norm();
            expect_equal_pcs(expanded_pcs, res2.pcs);

            res1.variance_explained.array() /= res1.total_variance;
            expect_equal_vectors(res1.variance_explained, res2.variance_explained);
        }

        // Comparing to a small variable cap, which has the same effect.
        {
            auto vrunner = runner;
            vrunner.set_block_weight_policy(scran::WeightPolicy::VARIABLE);
            vrunner.set_variable_block_weight_parameters({0, 0});
            auto vres = vrunner.run(com.get(), block2.data());

            vres.pcs.array() /= vres.pcs.norm();
            expect_equal_pcs(vres.pcs, res2.pcs);

            vres.variance_explained.array() /= vres.total_variance;
            expect_equal_vectors(vres.variance_explained, res2.variance_explained);
        }
    }

    // With a large size cap, each block is weighted by its size,
    // which is equivalent to the total absence of re-weighting.
    runner.set_block_weight_policy(scran::WeightPolicy::VARIABLE);
    runner.set_variable_block_weight_parameters({0, 1000000});
    {
        auto res2 = runner.run(com.get(), block2.data());

        scran::SimplePca runner2;
        runner2.set_scale(scale).set_rank(rank);
        auto ref2 = runner2.run(com.get());

        ref2.pcs.array() /= ref2.pcs.norm();
        res2.pcs.array() /= res2.pcs.norm();
        expect_equal_pcs(ref2.pcs, res2.pcs);

        ref2.variance_explained.array() /= ref2.total_variance;
        res2.variance_explained.array() /= res2.total_variance;
        expect_equal_vectors(ref2.variance_explained, res2.variance_explained);
    }
}

TEST_P(MultiBatchPcaMoreTest, ResidualOnly_VersusReference) {
    assemble(GetParam());
    auto block = generate_blocks(dense_row->ncol(), nblocks);

    scran::MultiBatchPca runner;
    runner.set_scale(scale).set_rank(rank);
    runner.set_use_residuals(true);
    runner.set_block_weight_policy(scran::WeightPolicy::NONE);

    auto res = runner.run(dense_row.get(), block.data());

    scran::ResidualPca refrunner;
    refrunner.set_scale(scale).set_rank(rank);
    refrunner.set_block_weight_policy(scran::WeightPolicy::NONE);

    auto ref = refrunner.run(dense_row.get(), block.data());

    expect_equal_vectors(res.variance_explained, ref.variance_explained);
    EXPECT_FLOAT_EQ(res.total_variance, ref.total_variance);
    expect_equal_rotation(res.rotation, ref.rotation);

    are_pcs_centered(res.pcs);
    EXPECT_EQ(res.pcs.rows(), rank);
    EXPECT_EQ(res.pcs.cols(), dense_row->ncol());
}

TEST_P(MultiBatchPcaMoreTest, WeightedResidual_VersusReference) {
    assemble(GetParam());
    auto block = generate_blocks(dense_row->ncol(), nblocks);

    scran::MultiBatchPca runner;
    runner.set_scale(scale).set_rank(rank);
    runner.set_use_residuals(true);
    runner.set_block_weight_policy(scran::WeightPolicy::EQUAL);

    auto res = runner.run(dense_row.get(), block.data());

    scran::ResidualPca refrunner;
    refrunner.set_scale(scale).set_rank(rank);
    refrunner.set_block_weight_policy(scran::WeightPolicy::EQUAL);

    auto ref = refrunner.run(dense_row.get(), block.data());

    expect_equal_vectors(res.variance_explained, ref.variance_explained);
    EXPECT_FLOAT_EQ(res.total_variance, ref.total_variance);
    expect_equal_rotation(res.rotation, ref.rotation);

    are_pcs_centered(res.pcs);
    EXPECT_EQ(res.pcs.rows(), rank);
    EXPECT_EQ(res.pcs.cols(), dense_row->ncol());
}

TEST_P(MultiBatchPcaMoreTest, WeightedResidual_DuplicatedBlocks) {
    assemble(GetParam());
    auto block = generate_blocks(dense_row->ncol(), nblocks);

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

    scran::MultiBatchPca runner;
    runner.set_scale(scale).set_rank(rank);
    runner.set_use_residuals(true);

    // Checking what happens if they're all equally weighted.
    runner.set_block_weight_policy(scran::WeightPolicy::EQUAL);
    {
        auto res2 = runner.run(com.get(), block2.data());
        res2.pcs.array() /= res2.pcs.norm();
        res2.variance_explained.array() /= res2.total_variance;

        {
            // Mocking up the expected results.
            auto res1 = runner.run(dense_row.get(), block.data());

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
            expanded_pcs.array() /= expanded_pcs.norm();
            expect_equal_pcs(expanded_pcs, res2.pcs);

            res1.variance_explained.array() /= res1.total_variance;
            expect_equal_vectors(res1.variance_explained, res2.variance_explained);
        }

        // Comparing to a small variable cap, which has the same effect.
        {
            auto vrunner = runner;
            vrunner.set_block_weight_policy(scran::WeightPolicy::VARIABLE);
            vrunner.set_variable_block_weight_parameters({0, 0});
            auto vres = vrunner.run(com.get(), block2.data());

            vres.pcs.array() /= vres.pcs.norm();
            expect_equal_pcs(vres.pcs, res2.pcs);

            vres.variance_explained.array() /= vres.total_variance;
            expect_equal_vectors(vres.variance_explained, res2.variance_explained);
        }
    }

    // With a large size cap, each block is weighted by its size,
    // which is equivalent to the total absence of re-weighting.
    runner.set_block_weight_policy(scran::WeightPolicy::VARIABLE);
    runner.set_variable_block_weight_parameters({0, 1000000});
    {
        auto res2 = runner.run(com.get(), block2.data());

        auto nrunner = runner;
        nrunner.set_block_weight_policy(scran::WeightPolicy::NONE);

        auto ref2 = nrunner.run(com.get(), block2.data());

        ref2.pcs.array() /= ref2.pcs.norm();
        res2.pcs.array() /= res2.pcs.norm();
        expect_equal_pcs(ref2.pcs, res2.pcs);

        ref2.variance_explained.array() /= ref2.total_variance;
        res2.variance_explained.array() /= res2.total_variance;
        expect_equal_vectors(ref2.variance_explained, res2.variance_explained);
    }
}

TEST_P(MultiBatchPcaMoreTest, SubsetTest) {
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

    scran::MultiBatchPca runner;
    runner.set_scale(scale).set_rank(rank);
    runner.set_block_weight_policy(scran::WeightPolicy::EQUAL);

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
    MultiBatchPca,
    MultiBatchPcaMoreTest,
    ::testing::Combine(
        ::testing::Values(false, true), // to scale or not to scale?
        ::testing::Values(2, 3, 4), // number of PCs to obtain
        ::testing::Values(1, 2, 3) // number of blocks
    )
);

/******************************************/

TEST(MultiBatchPcaTest, ReturnValues) {
    size_t nr = 120, nc = 131;
    auto mat = Simulator().matrix(nr, nc);
    auto block = generate_blocks(nc, 3);

    scran::MultiBatchPca runner;
    runner.set_rank(4);
    {
        auto ref = runner.run(&mat, block.data());
        EXPECT_EQ(ref.rotation.cols(), 0);
        EXPECT_EQ(ref.rotation.rows(), 0);
        EXPECT_EQ(ref.center.rows(), 0);
        EXPECT_EQ(ref.center.cols(), 0);
        EXPECT_EQ(ref.scale.size(), 0);
    }

    runner.set_return_center(true).set_return_scale(true).set_return_rotation(true);
    {
        auto ref = runner.run(&mat, block.data());
        EXPECT_EQ(ref.rotation.cols(), 4);
        EXPECT_EQ(ref.rotation.rows(), nr);
        EXPECT_EQ(ref.center.cols(), 1);
        EXPECT_EQ(ref.center.rows(), nr);
        EXPECT_EQ(ref.scale.size(), nr);
    }

    // Correct handling of center/scale with residuals.
    runner.set_use_residuals(true);
    {
        auto ref = runner.run(&mat, block.data());
        EXPECT_EQ(ref.rotation.cols(), 4);
        EXPECT_EQ(ref.rotation.rows(), nr);
        EXPECT_EQ(ref.center.cols(), 3);
        EXPECT_EQ(ref.center.rows(), nr);
        EXPECT_EQ(ref.scale.size(), nr);
    }
}
