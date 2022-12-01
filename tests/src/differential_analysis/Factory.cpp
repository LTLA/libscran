#include <gtest/gtest.h>

#include "../utils/macros.h"

#include "scran/differential_analysis/Factory.hpp"
#include "tatami/tatami.hpp"

#include "utils.h"
#include "../utils/compare_almost_equal.h"

/*******************************************************/

class DifferentialAnalysisEffectsCalculatorEdgeCaseTest : 
    public DifferentialAnalysisTestCore,
    public ::testing::Test
{};

TEST_F(DifferentialAnalysisEffectsCalculatorEdgeCaseTest, Simple) {
    size_t ncols = 18;
    size_t ngroups = 5;
    auto groups = create_groupings(ncols, ngroups);

    std::vector<double> mat_buffer;
    size_t nrows = 9;
    for (size_t r = 1; r <= nrows; ++r) {
        for (size_t c = 0; c < ncols; ++c) {
            mat_buffer.push_back(groups[c] * r);
        }
    }
    tatami::DenseRowMatrix<double, int> mat(nrows, ncols, std::move(mat_buffer));

    scran::differential_analysis::EffectsCalculator runner(1, 0);
    EffectsOverlord ova(true, nrows, ngroups);
    auto state = runner.run(&mat, groups.data(), ngroups, ova);

    // Check that the means are as expected.
    for (size_t r = 0; r < nrows; ++r) {
        auto it = state.means.begin() + r * ngroups;
        std::vector<double> observed(it, it + ngroups);

        std::vector<double> expected(ngroups);
        for (size_t g = 0; g < ngroups; ++g) {
            expected[g] = g * (r + 1);
        }
        EXPECT_EQ(expected, observed);
    }

    // Check that the variances and detected are as expected.
    EXPECT_EQ(state.variances, std::vector<double>(nrows * ngroups));

    std::vector<double> expected(nrows * ngroups, 1);
    for (size_t r = 0; r < nrows; ++r) {
        expected[r * ngroups] = 0; // first group is zero because the group index is zero.
    }
    EXPECT_EQ(state.detected, expected);

    // Check that the pairwise AUCs are as expected.
    for (size_t r = 0; r < nrows; ++r) {
        auto base = ova.store.begin() + r * ngroups * ngroups;
        std::vector<double> observed(base, base + ngroups * ngroups);

        std::vector<double> expected(ngroups * ngroups);
        for (size_t g1 = 0; g1 < ngroups; ++g1) {
            for (size_t g2 = 0; g2 < ngroups; ++g2) {
                if (g1 == g2) {
                    continue;
                }
                expected[g1 * ngroups + g2] = g1 > g2;
            }
        }

        EXPECT_EQ(observed, expected);
    }
}

TEST_F(DifferentialAnalysisEffectsCalculatorEdgeCaseTest, AllZeros) {
    size_t ncols = 21;
    size_t ngroups = 5;
    auto groups = create_groupings(ncols, ngroups);

    size_t nrows = 3;
    std::vector<double> mat_buffer(ncols * nrows);
    tatami::DenseRowMatrix<double, int> mat(nrows, ncols, std::move(mat_buffer));

    scran::differential_analysis::EffectsCalculator runner(1, 0);
    EffectsOverlord ova(true, nrows, ngroups);
    auto state = runner.run(&mat, groups.data(), ngroups, ova);

    EXPECT_EQ(state.means, std::vector<double>(nrows * ngroups));
    EXPECT_EQ(state.variances, std::vector<double>(nrows * ngroups));
    EXPECT_EQ(state.detected, std::vector<double>(nrows * ngroups));

    // AUCs should all be tied if they're all zero.
    std::vector<double> expected(nrows * ngroups * ngroups, 0.5);
    for (size_t r = 0; r < nrows; ++r) {
        for (size_t g = 0; g < ngroups; ++g) {
            expected[r * ngroups * ngroups + g * ngroups + g] = 0; 
        }
    }
    EXPECT_EQ(ova.store, expected);
}

TEST_F(DifferentialAnalysisEffectsCalculatorEdgeCaseTest, MissingGroup) {
    size_t ncols = 17;
    size_t ngroups = 3;

    auto groups = create_groupings(ncols, ngroups);
    int lost = 1;
    for (auto& g : groups) {
        if (g == lost) {
            g = 0;
        }
    }

    size_t nrows = 13;
    assemble(nrows, ncols);

    scran::differential_analysis::EffectsCalculator runner(1, 0);
    EffectsOverlord ova(true, nrows, ngroups);
    auto state = runner.run(dense_row->get(), groups.data(), ngroups, ova);

    for (size_t r = 0; r < nrows; ++r) {
        EXPECT_TRUE(std::isnan(state.means[r * ngroups + lost]));
        EXPECT_TRUE(std::isnan(state.variances[r * ngroups + lost]));
        EXPECT_TRUE(std::isnan(state.detected[r * ngroups + lost]));
    }
}

TEST_F(DifferentialAnalysisEffectsCalculatorEdgeCaseTest, MissingBlock) {
    size_t ncols = 19;
    size_t ngroups = 3;
    auto groups = create_groupings(ncols, ngroups);

    int nblocks = 2;
    std::vector<int> blocks(ncols, 1);

    size_t nrows = 21;
    assemble(nrows, ncols);

    scran::differential_analysis::EffectsCalculator runner(1, 0);
    EffectsOverlord ova(true, nrows, ngroups);
    auto state = runner.run_blocked(dense_row->get(), groups.data(), ngroups, blocks.data(), nblocks, ova);

    for (size_t r = 0; r < nrows; ++r) {
        for (int g = 0; g < ngroups; ++g) {
            EXPECT_TRUE(std::isnan(state.means[r * ngroups * nblocks + g]));
            EXPECT_TRUE(std::isnan(state.variances[r * ngroups * nblocks + g]));
            EXPECT_TRUE(std::isnan(state.detected[r * ngroups * nblocks + g]));

            EXPECT_FALSE(std::isnan(state.means[r * ngroups * nblocks + ngroups + g]));
            EXPECT_FALSE(std::isnan(state.variances[r * ngroups * nblocks + ngroups + g]));
            EXPECT_FALSE(std::isnan(state.detected[r * ngroups * nblocks + ngroups + g]));
        }
    }
}

/*******************************************************/

class DifferentialAnalysisEffectsCalculatorUnblockedTest : 
    public DifferentialAnalysisTestCore,
    public ::testing::TestWithParam<std::tuple<int, bool, int> >
{
protected:
    static constexpr size_t nrows = 100;
    static constexpr size_t ncols = 50;

    void SetUp() {
        assemble(nrows, ncols);
    }
};

TEST_P(DifferentialAnalysisEffectsCalculatorUnblockedTest, Manual) {
    auto param = GetParam();
    auto ngroups = std::get<0>(param);
    bool do_auc = std::get<1>(param);
    auto nthreads = std::get<2>(param);

    auto groups = create_groupings(ncols, ngroups);

    scran::differential_analysis::EffectsCalculator runner(nthreads, 0);
    EffectsOverlord ova(do_auc, nrows, ngroups);
    auto state = runner.run(dense_row.get(), groups.data(), ngroups, ova);

    // Manually calculating the mean and variance.
    std::vector<double> means(nrows * ngroups), detected(nrows * ngroups);
    std::vector<double> aucs(do_auc ? nrows * ngroups * ngroups : 0);

    for (size_t g = 0; g < nrows; ++g) {
        std::vector<double> buf(ncols);
        auto ptr = dense_row->row(g, buf.data());

        for (int c = 0; c < buf.size(); ++c) {
            size_t offset = g * ngroups + groups[c];
            means[offset] += ptr[c];
            detected[offset] += (ptr[c] != 0);
        }

        for (int b = 0; b < ngroups; ++b) {
            size_t offset = g * ngroups + b;
            means[offset] /= state.level_size[b];
            detected[offset] /= state.level_size[b];
        }

        if (do_auc) {
            // Manually computing the AUCs.
            scran::differential_analysis::PairedStore store;
            std::vector<int> num_zeros(ngroups);
            for (size_t c = 0; c < ncols; ++c) {
                if (ptr[c]) {
                    store.emplace_back(ptr[c], groups[c]);
                } else {
                    ++num_zeros[groups[c]];
                }
            }
            scran::differential_analysis::compute_pairwise_auc(store, num_zeros, state.level_size, aucs.data() + g * ngroups * ngroups, true);
        }
    }

    EXPECT_EQ(state.means, means);
    EXPECT_EQ(state.detected, detected);
    if (do_auc) {
        EXPECT_EQ(aucs, ova.store);
    }
}

TEST_P(DifferentialAnalysisEffectsCalculatorUnblockedTest, Consistency) {
    auto param = GetParam();
    auto ngroups = std::get<0>(param);
    bool do_auc = std::get<1>(param);
    auto nthreads = std::get<2>(param);

    auto groups = create_groupings(ncols, ngroups);

    scran::differential_analysis::EffectsCalculator runner(nthreads, 0);
    EffectsOverlord ova(do_auc, nrows, ngroups);
    auto state = runner.run(dense_row.get(), groups.data(), ngroups, ova);

    // Checking the various helper statistics.
    EXPECT_EQ(state.ngroups, ngroups);
    EXPECT_EQ(state.nblocks, 1);
    EXPECT_EQ(state.ngenes, nrows);

    std::vector<int> num_per_group(ngroups);
    for (size_t i = 0; i < ncols; ++i) {
        ++num_per_group[groups[i]];
    }
    EXPECT_EQ(num_per_group, state.level_size);

    // Checking aganst the single-core case.
    if (nthreads > 1) {
        scran::differential_analysis::EffectsCalculator runner(1, 0);
        EffectsOverlord ova1(do_auc, nrows, ngroups);
        auto state1 = runner.run(dense_row.get(), groups.data(), ngroups, ova1);
        EXPECT_EQ(state.means, state1.means);
        EXPECT_EQ(state.variances, state1.variances);
        EXPECT_EQ(state.detected, state1.detected);
        EXPECT_EQ(ova.store, ova1.store);
    }

    // Comparing different implementations that are used for different
    // matrices. This is our version of a reference check, otherwise we'd just
    // be reimplementing one of the implementations, and that's no fun.
    EffectsOverlord ova2(do_auc, nrows, ngroups);
    auto state2 = runner.run(sparse_row.get(), groups.data(), ngroups, ova2);
    compare_almost_equal(state.means, state2.means);
    compare_almost_equal(state.variances, state2.variances);
    EXPECT_EQ(state.detected, state2.detected);

    EffectsOverlord ova3(do_auc, nrows, ngroups);
    auto state3 = runner.run(dense_column.get(), groups.data(), ngroups, ova3);
    compare_almost_equal(state.means, state3.means);
    compare_almost_equal(state.variances, state3.variances);
    EXPECT_EQ(state.detected, state3.detected);

    EffectsOverlord ova4(do_auc, nrows, ngroups);
    auto state4 = runner.run(sparse_column.get(), groups.data(), ngroups, ova4);
    compare_almost_equal(state.means, state4.means);
    compare_almost_equal(state.variances, state4.variances);
    EXPECT_EQ(state.detected, state4.detected);

    // Checking AUCs.
    EXPECT_EQ(ova.store, ova2.store);
    EXPECT_EQ(ova.store, ova3.store);
    EXPECT_EQ(ova.store, ova4.store);
}

TEST_P(DifferentialAnalysisEffectsCalculatorUnblockedTest, SingleBlock) {
    auto param = GetParam();
    auto ngroups = std::get<0>(param);
    bool do_auc = std::get<1>(param);
    auto nthreads = std::get<2>(param);

    auto groups = create_groupings(ncols, ngroups);

    scran::differential_analysis::EffectsCalculator runner(nthreads, 0);
    EffectsOverlord ova(do_auc, nrows, ngroups);
    auto state = runner.run(dense_row.get(), groups.data(), ngroups, ova);

    // Checking that we get the same results.
    auto state2 = runner.run_blocked(dense_row.get(), groups.data(), ngroups, static_cast<const int*>(NULL), 1, ova);
    EXPECT_EQ(state.level_size, state2.level_size);
    EXPECT_EQ(state.means, state2.means);
    EXPECT_EQ(state.variances, state2.variances);
    EXPECT_EQ(state.detected, state2.detected);

    std::vector<int> block(ncols);
    auto state3 = runner.run_blocked(dense_row.get(), groups.data(), ngroups, block.data(), 1, ova);
    EXPECT_EQ(state.level_size, state3.level_size);
    EXPECT_EQ(state.means, state3.means);
    EXPECT_EQ(state.variances, state3.variances);
    EXPECT_EQ(state.detected, state3.detected);
}

INSTANTIATE_TEST_CASE_P(
    DifferentialAnalysisEffectsCalculatorUnblocked,
    DifferentialAnalysisEffectsCalculatorUnblockedTest,
    ::testing::Combine(
        ::testing::Values(2, 3, 5, 7), // number of clusters
        ::testing::Values(false, true), // with or without the AUC?
        ::testing::Values(1, 3) // number of threads
    )
);

/*******************************************************/

class DifferentialAnalysisEffectsCalculatorBlockedTest : 
    public DifferentialAnalysisTestCore,
    public ::testing::TestWithParam<std::tuple<int, int, bool, int> >
{
protected:
    static constexpr size_t nrows = 100;
    static constexpr size_t ncols = 50;

    void SetUp() {
        assemble(nrows, ncols);
    }
};

TEST_P(DifferentialAnalysisEffectsCalculatorBlockedTest, Consistency) {
    auto param = GetParam();
    auto ngroups = std::get<0>(param);
    auto nblocks = std::get<1>(param);
    bool do_auc = std::get<2>(param);
    auto nthreads = std::get<3>(param);

    auto groups = create_groupings(ncols, ngroups);
    auto blocks = create_blocks(ncols, nblocks);

    scran::differential_analysis::EffectsCalculator runner(nthreads, 0);
    EffectsOverlord ova(do_auc, nrows, ngroups);
    auto state = runner.run_blocked(dense_row.get(), groups.data(), ngroups, blocks.data(), nblocks, ova);

    // Checking the various helper statistics.
    EXPECT_EQ(state.ngroups, ngroups);
    EXPECT_EQ(state.nblocks, nblocks);
    EXPECT_EQ(state.ngenes, nrows);

    std::vector<int> num_per_group(ngroups * nblocks);
    for (size_t i = 0; i < ncols; ++i) {
        ++num_per_group[groups[i] * nblocks + blocks[i]];
    }
    EXPECT_EQ(num_per_group, state.level_size);

    // Checking aganst the single-core case.
    if (nthreads > 1) {
        scran::differential_analysis::EffectsCalculator runner(1, 0);
        EffectsOverlord ova1(do_auc, nrows, ngroups);
        auto state1 = runner.run_blocked(dense_row.get(), groups.data(), ngroups, blocks.data(), nblocks, ova1);
        EXPECT_EQ(state.means, state1.means);
        EXPECT_EQ(state.variances, state1.variances);
        EXPECT_EQ(state.detected, state1.detected);
        EXPECT_EQ(ova.store, ova1.store);
    }

    // Comparing different implementations that are used for different
    // matrices. This is our version of a reference check, otherwise we'd just
    // be reimplementing one of the implementations, and that's no fun.
    EffectsOverlord ova2(do_auc, nrows, ngroups);
    auto state2 = runner.run_blocked(sparse_row.get(), groups.data(), ngroups, blocks.data(), nblocks, ova2);
    compare_almost_equal(state.means, state2.means);
    compare_almost_equal(state.variances, state2.variances);
    EXPECT_EQ(state.detected, state2.detected);

    EffectsOverlord ova3(do_auc, nrows, ngroups);
    auto state3 = runner.run_blocked(dense_column.get(), groups.data(), ngroups, blocks.data(), nblocks, ova3);
    compare_almost_equal(state.means, state3.means);
    compare_almost_equal(state.variances, state3.variances);
    EXPECT_EQ(state.detected, state3.detected);

    EffectsOverlord ova4(do_auc, nrows, ngroups);
    auto state4 = runner.run_blocked(sparse_column.get(), groups.data(), ngroups, blocks.data(), nblocks, ova4);
    compare_almost_equal(state.means, state4.means);
    compare_almost_equal(state.variances, state4.variances);
    EXPECT_EQ(state.detected, state4.detected);

    // Checking AUCs.
    EXPECT_EQ(ova.store, ova2.store);
    EXPECT_EQ(ova.store, ova3.store);
    EXPECT_EQ(ova.store, ova4.store);
}

TEST_P(DifferentialAnalysisEffectsCalculatorBlockedTest, Subsetted) {
    auto param = GetParam();
    auto ngroups = std::get<0>(param);
    auto nblocks = std::get<1>(param);
    bool do_auc = std::get<2>(param);
    auto nthreads = std::get<3>(param);

    auto groups = create_groupings(ncols, ngroups);
    auto blocks = create_blocks(ncols, nblocks);

    scran::differential_analysis::EffectsCalculator runner(nthreads, 0);

    std::vector<double> means(ngroups * nblocks * nrows);
    std::vector<double> variances(means.size());
    std::vector<double> detected(means.size());

    for (int b = 0; b < nblocks; ++b) {
        std::vector<int> subset;
        std::vector<int> subgroups;
        for (int i = 0; i < ncols; ++i) {
            if (blocks[i] == b) {
                subset.push_back(i);
                subgroups.push_back(groups[i]);
            }
        }

        auto sub = tatami::make_DelayedSubset<1>(dense_row, std::move(subset));
        EffectsOverlord subova(do_auc, nrows, ngroups);
        auto substate = runner.run(sub.get(), subgroups.data(), ngroups, subova);

        for (size_t r = 0; r < nrows; ++r) {
            size_t offset = r * ngroups;
            for (int g = 0; g < ngroups; ++g) {
                size_t outset = r * (ngroups * nblocks) + g * nblocks + b;
                means[outset] = substate.means[offset + g];
                variances[outset] = substate.variances[offset + g];
                detected[outset] = substate.detected[offset + g];
            }
        }
    }

    EffectsOverlord ova(do_auc, nrows, ngroups);
    auto state = runner.run_blocked(dense_row.get(), groups.data(), ngroups, blocks.data(), nblocks, ova);
    EXPECT_EQ(means, state.means);
    EXPECT_EQ(variances, state.variances);
    EXPECT_EQ(detected, state.detected);
}

TEST_P(DifferentialAnalysisEffectsCalculatorBlockedTest, Duplicates) {
    auto param = GetParam();
    auto ngroups = std::get<0>(param);
    auto nblocks = std::get<1>(param);
    bool do_auc = std::get<2>(param);
    auto nthreads = std::get<3>(param);

    auto groups0 = create_groupings(ncols, ngroups);
    scran::differential_analysis::EffectsCalculator runner(nthreads, 0);
    EffectsOverlord ova(do_auc, nrows, ngroups);
    auto ref = runner.run(dense_row.get(), groups0.data(), ngroups, ova);

    // Duplicate the matrix and check that we get the same AUC results.
    std::vector<int> blocks, groups;
    std::vector<std::shared_ptr<tatami::NumericMatrix> > bound;
    for (int b = 0; b < nblocks; ++b) {
        bound.push_back(dense_row);
        groups.insert(groups.end(), groups0.begin(), groups0.end());
        blocks.resize(blocks.size() + dense_row->ncol(), b);
    }
    auto mat = tatami::make_DelayedBind<1>(std::move(bound));

    EffectsOverlord ova2(do_auc, nrows, ngroups);
    auto state2 = runner.run_blocked(mat.get(), groups.data(), ngroups, blocks.data(), nblocks, ova2);
    EXPECT_EQ(state2.ngroups, ngroups);
    EXPECT_EQ(state2.nblocks, nblocks);
    EXPECT_EQ(ova.store, ova2.store);

    // Also checking the various statistics.
    std::vector<double> means(ngroups * nblocks * nrows);
    std::vector<double> variances(means.size());
    std::vector<double> detected(means.size());
    for (size_t r = 0; r < nrows; ++r) {
        size_t offset = r * ngroups;
        for (int g = 0; g < ngroups; ++g) {
            size_t outset = r * (ngroups * nblocks) + g * nblocks;
            std::fill(means.begin() + outset, means.begin() + outset + nblocks, ref.means[offset + g]);
            std::fill(variances.begin() + outset, variances.begin() + outset + nblocks, ref.variances[offset + g]);
            std::fill(detected.begin() + outset, detected.begin() + outset + nblocks, ref.detected[offset + g]);
        }
    }

    EXPECT_EQ(means, state2.means);
    EXPECT_EQ(variances, state2.variances);
    EXPECT_EQ(detected, state2.detected);
}

INSTANTIATE_TEST_CASE_P(
    DifferentialAnalysisEffectsCalculatorBlocked,
    DifferentialAnalysisEffectsCalculatorBlockedTest,
    ::testing::Combine(
        ::testing::Values(2, 3, 5), // number of clusters
        ::testing::Values(2, 3), // number of blocks
        ::testing::Values(false, true), // with or without the AUC?
        ::testing::Values(1, 3) // number of threads
    )
);

/*******************************************************/

class DifferentialAnalysisEffectsCalculatorThresholdTest : 
    public DifferentialAnalysisTestCore,
    public ::testing::Test
{
protected:
    static constexpr size_t nrows = 100;
    static constexpr size_t ncols = 50;

    void SetUp() {
        assemble(nrows, ncols);
    }
};

TEST_F(DifferentialAnalysisEffectsCalculatorThresholdTest, Smaller) {
    int ngroups = 2;
    auto groups = create_groupings(ncols, ngroups);

    scran::differential_analysis::EffectsCalculator runner0(1, 0);
    EffectsOverlord ova0(true, nrows, ngroups);
    auto state0 = runner0.run(dense_row.get(), groups.data(), ngroups, ova0);

    scran::differential_analysis::EffectsCalculator runner1(1, 1);
    EffectsOverlord ova1(true, nrows, ngroups);
    auto state1 = runner1.run(dense_row.get(), groups.data(), ngroups, ova1);

    // Setting a threshold should always result in a smaller AUC - 
    // not necessarily an AUC closer to 0.5, just smaller, because
    // the treatment of a threshold is as if the distribution was shifted.
    for (size_t r = 0; r < nrows; ++r) {
        for (int g1 = 0; g1 < ngroups; ++g1) {
            for (int g2 = 0; g2 < ngroups; ++g2) {
                if (g1 == g2) {
                    continue;
                }
                size_t offset = r * ngroups * ngroups + g1 * ngroups + g2;
                EXPECT_TRUE(ova0.store[offset] >= ova1.store[offset]);
            }
        }
    }

    // Checking that at least one entry was a '>'.
    EXPECT_NE(ova0.store, ova1.store);
}

TEST_F(DifferentialAnalysisEffectsCalculatorThresholdTest, Zeroed) {
    int ngroups = 2;
    auto groups = create_groupings(ncols, ngroups);

    // Using a huge threshold should force all AUCs to zero.
    scran::differential_analysis::EffectsCalculator runner(1, 1000);
    EffectsOverlord ova(true, nrows, ngroups);
    auto state = runner.run(dense_row.get(), groups.data(), ngroups, ova);
    EXPECT_EQ(ova.store, std::vector<double>(ngroups * ngroups * nrows));
}
