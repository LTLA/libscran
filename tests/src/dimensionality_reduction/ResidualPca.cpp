#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "../data/Simulator.hpp"
#include "compare_pcs.h"

#include "tatami/tatami.hpp"

#include "scran/dimensionality_reduction/ResidualPca.hpp"
#include "scran/dimensionality_reduction/SimplePca.hpp"

TEST(RegressWrapperTest, EigenDense) {
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

    scran::pca_utils::RegressWrapper<decltype(thing), int> blocked(&thing, block.data(), &centers);
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

TEST(RegressWrapperTest, CustomSparse) {
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
    scran::pca_utils::RegressWrapper<decltype(thing), int> blocked(&thing, block.data(), &centers);
    auto realized = blocked.realize();

    // Checking that the dense reference matches up.
    {
        scran::pca_utils::RegressWrapper<decltype(ref), int> blockedref(&ref, block.data(), &centers);
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

class ResidualPcaTestCore {
protected:
    std::shared_ptr<tatami::NumericMatrix> dense_row;

    template<class Param>
    void assemble(const Param& param) {
        scale = std::get<0>(param);
        rank = std::get<1>(param);
        nblocks = std::get<2>(param);

        size_t nr = 121, nc = 155;
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

class ResidualPcaBasicTest : public ::testing::TestWithParam<std::tuple<bool, int, int, int> >, public ResidualPcaTestCore {
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

TEST_P(ResidualPcaBasicTest, BasicConsistency) {
    auto param = GetParam();
    extra_assemble(param);
    int nthreads = std::get<3>(param);

    scran::ResidualPca runner;
    runner.set_scale(scale).set_rank(rank);
    runner.set_block_weight_policy(scran::WeightPolicy::NONE);
    auto block = generate_blocks(dense_row->ncol(), nblocks);
    auto ref = runner.run(dense_row.get(), block.data());

    if (nthreads == 1) {
        EXPECT_EQ(ref.pcs.rows(), rank);
        EXPECT_EQ(ref.pcs.cols(), dense_row->ncol());
        EXPECT_EQ(ref.variance_explained.size(), rank);

        are_pcs_centered(ref.pcs);
        EXPECT_TRUE(ref.total_variance >= std::accumulate(ref.variance_explained.begin(), ref.variance_explained.end(), 0.0));

        // Total variance makes sense. Remember, this doesn't consider the
        // loss of d.f. from calculation of the block means.
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

TEST_P(ResidualPcaBasicTest, WeightedConsistency) {
    auto param = GetParam();
    extra_assemble(param);
    int nthreads = std::get<3>(param);

    scran::ResidualPca runner;
    runner.set_scale(scale).set_rank(rank);
    runner.set_block_weight_policy(scran::WeightPolicy::EQUAL);

    auto block = generate_blocks(dense_row->ncol(), nblocks);
    auto ref = runner.run(dense_row.get(), block.data());

    if (nthreads == 1) {
        are_pcs_centered(ref.pcs);
        EXPECT_TRUE(ref.total_variance >= std::accumulate(ref.variance_explained.begin(), ref.variance_explained.end(), 0.0));

        if (scale) {
            EXPECT_FLOAT_EQ(dense_row->nrow(), ref.total_variance);
        } else {
            auto collected = fragment_matrices_by_block(dense_row, block, nblocks);

            // Here, the 'variance' is really just the grand sum (across blocks) of
            // the sum (across cells) of the squared difference from the mean.
            double total_var = 0;
            for (int b = 0, end = collected.size(); b < end; ++b) {
                const auto& sub = collected[b];
                auto vars = tatami::row_variances(sub.get());
                total_var += std::accumulate(vars.begin(), vars.end(), 0.0) * (sub->ncol() - 1) / sub->ncol();
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
    ResidualPca,
    ResidualPcaBasicTest,
    ::testing::Combine(
        ::testing::Values(false, true), // to scale or not to scale?
        ::testing::Values(2, 3, 4), // number of PCs to obtain
        ::testing::Values(1, 2, 3), // number of blocks
        ::testing::Values(1, 3) // number of threads
    )
);

/******************************************/

class ResidualPcaMoreTest : public ::testing::TestWithParam<std::tuple<bool, int, int> >, public ResidualPcaTestCore {};

TEST_P(ResidualPcaMoreTest, VersusSimple) {
    assemble(GetParam());
    auto block = generate_blocks(dense_row->ncol(), nblocks);

    scran::ResidualPca runner;
    runner.set_scale(scale).set_rank(rank);
    runner.set_block_weight_policy(scran::WeightPolicy::NONE);
    auto res1 = runner.run(dense_row.get(), block.data());

    if (nblocks == 1) {
        // Checking that we get more-or-less the same results
        // from the vanilla PCA algorithm in the absence of blocks.
        scran::SimplePca ref;
        ref.set_scale(scale).set_rank(rank);
        auto res2 = ref.run(dense_row.get());

        expect_equal_pcs(res1.pcs, res2.pcs);
        expect_equal_vectors(res1.variance_explained, res2.variance_explained);
        EXPECT_FLOAT_EQ(res1.total_variance, res2.total_variance);
    } else {
        // Manually regressing things out.
        size_t nr = dense_row->nrow(), nc = dense_row->ncol();
        std::vector<double> regressed(nr * nc);
        for (int b = 0; b < nblocks; ++b) {
            std::vector<int> keep;
            for (size_t i = 0; i < block.size(); ++i) {
                if (block[i] == b) {
                    keep.push_back(i);
                }
            }

            if (keep.empty()) {
                continue;
            }

            auto sub = tatami::make_DelayedSubset<1>(dense_row, keep);
            auto center = tatami::row_sums(sub.get());
            for (auto& x : center) {
                x /= keep.size();
            }

            auto ext = dense_row->dense_column();
            for (auto i : keep) {
                auto ptr = regressed.data() + i * static_cast<size_t>(nr);
                ext->fetch_copy(i, ptr);
                for (auto x : center) {
                    *ptr -= x;
                    ++ptr;
                }
            }
        }

        scran::SimplePca ref;
        ref.set_scale(scale).set_rank(rank);

        tatami::DenseColumnMatrix<double> refmat(nr, nc, std::move(regressed));
        auto res2 = ref.run(&refmat);

        expect_equal_pcs(res1.pcs, res2.pcs);
        expect_equal_vectors(res1.variance_explained, res2.variance_explained);
        EXPECT_FLOAT_EQ(res1.total_variance, res2.total_variance);
    }
}

TEST_P(ResidualPcaMoreTest, SubsetTest) {
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

    scran::ResidualPca runner;
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
    ResidualPca,
    ResidualPcaMoreTest,
    ::testing::Combine(
        ::testing::Values(false, true), // to scale or not to scale?
        ::testing::Values(2, 3, 4), // number of PCs to obtain
        ::testing::Values(1, 2, 3) // number of blocks
    )
);

/******************************************/

class ResidualPcaWeightedTest : public ::testing::TestWithParam<std::tuple<bool, int, int, int> > {
protected:
    std::vector<std::shared_ptr<tatami::NumericMatrix> > components;
    std::vector<int> blocking;

    template<class Param>
    void assemble(Param param) {
        scale = std::get<0>(param);
        rank = std::get<1>(param);
        nblocks = std::get<2>(param);

        size_t nr = 50, nc = 20;
        Simulator sim;
        for (int b = 0; b < nblocks; ++b) {
            sim.seed += b; // slightly different seed to keep things interesting.
            auto mat = sim.matrix(nr, nc);
            components.emplace_back(new decltype(mat)(std::move(mat)));
            blocking.insert(blocking.end(), nc, b);
        }
    }

    bool scale;
    int rank;
    int nblocks;
};

TEST_P(ResidualPcaWeightedTest, VersusReference) {
    auto param = GetParam();
    assemble(param);
    int nthreads = std::get<3>(param);

    scran::ResidualPca runner;
    runner.set_scale(scale).set_rank(rank);
    runner.set_block_weight_policy(scran::WeightPolicy::NONE);
    auto combined = tatami::make_DelayedBind<1>(components);
    auto ref = runner.run(combined.get(), blocking.data());

    // Some adjustment is required to adjust for the global scaling.
    ref.pcs.array() /= ref.pcs.norm();
    ref.variance_explained.array() /= ref.total_variance;

    // Checking that we get more-or-less the same results with weighting.
    runner.set_num_threads(nthreads);
    runner.set_block_weight_policy(scran::WeightPolicy::EQUAL);

    auto res1 = runner.run(combined.get(), blocking.data());
    res1.pcs.array() /= res1.pcs.norm();
    expect_equal_pcs(ref.pcs, res1.pcs);

    res1.variance_explained.array() /= res1.total_variance;
    expect_equal_vectors(ref.variance_explained, res1.variance_explained);

    // Manually adding more instances of a block.
    auto expanded_block = blocking;
    for (int b = 0; b < nblocks; ++b) {
        for (int b0 = 0; b0 < b; ++b0) {
            components.push_back(components[b]);
        }
        expanded_block.insert(expanded_block.end(), b * components[b]->ncol(), b);
    }
    auto expanded = tatami::make_DelayedBind<1>(components);

    // With a large size cap, each block is weighted by its size,
    // which is equivalent to the total absence of re-weighting.
    runner.set_block_weight_policy(scran::WeightPolicy::VARIABLE);
    runner.set_variable_block_weight_parameters({ 0, 1000000 });
    {
        auto res2 = runner.run(expanded.get(), expanded_block.data());

        scran::ResidualPca runner2;
        runner2.set_scale(scale).set_rank(rank);
        runner2.set_block_weight_policy(scran::WeightPolicy::NONE);
        auto ref2 = runner2.run(expanded.get(), expanded_block.data());

        ref2.pcs.array() /= ref2.pcs.norm();
        res2.pcs.array() /= res2.pcs.norm();
        expect_equal_pcs(ref2.pcs, res2.pcs);

        ref2.variance_explained.array() /= ref2.total_variance;
        res2.variance_explained.array() /= res2.total_variance;
        expect_equal_vectors(ref2.variance_explained, res2.variance_explained);
    }

    // We turn down the size cap so that every batch is equally weighted.
    runner.set_variable_block_weight_parameters({ 0, 0 });
    {
        auto res2 = runner.run(expanded.get(), expanded_block.data());

        // Mocking up the expected results.
        Eigen::MatrixXd expanded_pcs(rank, expanded->ncol());
        expanded_pcs.leftCols(combined->ncol()) = ref.pcs;
        size_t host_counter = 0, dest_counter = combined->ncol();
        for (int b = 0; b < nblocks; ++b) {
            size_t nc = components[b]->ncol();
            for (int b0 = 0; b0 < b; ++b0) {
                expanded_pcs.middleCols(dest_counter, nc) = ref.pcs.middleCols(host_counter, nc);
                dest_counter += nc; 
            }
            host_counter += nc;
        }

        Eigen::VectorXd recenters = expanded_pcs.rowwise().sum();
        recenters /= expanded_pcs.cols();
        for (size_t i = 0, end = expanded_pcs.cols(); i < end; ++i) {
            expanded_pcs.col(i) -= recenters;
        }

        // Comparing the results.
        expanded_pcs.array() /= expanded_pcs.norm();
        res2.pcs.array() /= res2.pcs.norm();
        expect_equal_pcs(expanded_pcs, res2.pcs);

        res2.variance_explained.array() /= res2.total_variance;
        expect_equal_vectors(ref.variance_explained, res2.variance_explained);
    }
}

INSTANTIATE_TEST_SUITE_P(
    ResidualPca,
    ResidualPcaWeightedTest,
    ::testing::Combine(
        ::testing::Values(false, true), // to scale or not to scale?
        ::testing::Values(2, 3, 4), // number of PCs to obtain
        ::testing::Values(2, 3), // number of blocks
        ::testing::Values(1, 3) // number of threads
    )
);

/******************************************/

TEST(ResidualPcaTest, ReturnValues) {
    size_t nr = 120, nc = 131;
    auto mat = Simulator().matrix(nr, nc);
    auto block = generate_blocks(nc, 3);

    scran::ResidualPca runner;
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
        EXPECT_EQ(ref.center.cols(), 3);
        EXPECT_EQ(ref.center.rows(), nr);
        EXPECT_EQ(ref.scale.size(), nr);
    }
}
