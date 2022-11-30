#include <gtest/gtest.h>
#include "../utils/macros.h"
#include "scran/differential_analysis/Factory.hpp"
#include "utils.h"
#include "../utils/compare_almost_equal.h"

class DifferentialAnalysisEffectsCalculatorTestCore : public DifferentialAnalysisTestCore {
protected:
    struct Overlord {
        Overlord(bool a, size_t nrows, int ngroups) : do_auc(a), store(nrows * ngroups * ngroups) {}

        bool do_auc;
        bool needs_auc() const {
            return do_auc;
        }

        std::vector<double> store;

        struct ComplexWorker {
            ComplexWorker(double* s) : store(s) {}

            std::vector<double> buffer;
            double* store;
            double* prepare_auc_buffer(size_t i, int ngroups) {
                buffer.resize(ngroups * ngroups);
                return buffer.data();
            }

            void consume_auc_buffer(size_t i, int ngroups, double*) {
                std::copy(buffer.begin(), buffer.end(), store + i * ngroups * ngroups);
                return;
            }
        };

        ComplexWorker complex_worker() {
            return ComplexWorker(store.data());
        }
    };
};

/*******************************************************/

class DifferentialAnalysisEffectsCalculatorEdgeCaseTest : 
    public DifferentialAnalysisEffectsCalculatorTestCore,
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
    Overlord ova(true, nrows, ngroups);
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
    Overlord ova(true, nrows, ngroups);
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

/*******************************************************/

class DifferentialAnalysisEffectsCalculatorUnblockedTest : 
    public DifferentialAnalysisEffectsCalculatorTestCore,
    public ::testing::TestWithParam<std::tuple<int, bool, int> >
{
protected:
    static constexpr size_t nrows = 100;
    static constexpr size_t ncols = 50;

    void SetUp() {
        assemble(nrows, ncols);
    }
};

TEST_P(DifferentialAnalysisEffectsCalculatorUnblockedTest, Unblocked) {
    auto param = GetParam();
    auto ngroups = std::get<0>(param);
    bool do_auc = std::get<1>(param);
    auto nthreads = std::get<2>(param);

    auto groups = create_groupings(ncols, ngroups);

    scran::differential_analysis::EffectsCalculator runner(nthreads, 0);
    Overlord ova(do_auc, nrows, ngroups);
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
        Overlord ova1(do_auc, nrows, ngroups);
        auto state1 = runner.run(dense_row.get(), groups.data(), ngroups, ova1);
        EXPECT_EQ(state.means, state1.means);
        EXPECT_EQ(state.variances, state1.variances);
        EXPECT_EQ(state.detected, state1.detected);
        EXPECT_EQ(ova.store, ova1.store);
    }

    // Comparing different implementations that are used for different
    // matrices. This is our version of a reference check, otherwise we'd just
    // be reimplementing one of the implementations, and that's no fun.
    Overlord ova2(do_auc, nrows, ngroups);
    auto state2 = runner.run(sparse_row.get(), groups.data(), ngroups, ova2);
    compare_almost_equal(state.means, state2.means);
    compare_almost_equal(state.variances, state2.variances);
    EXPECT_EQ(state.detected, state2.detected);

    Overlord ova3(do_auc, nrows, ngroups);
    auto state3 = runner.run(dense_column.get(), groups.data(), ngroups, ova3);
    compare_almost_equal(state.means, state3.means);
    compare_almost_equal(state.variances, state3.variances);
    EXPECT_EQ(state.detected, state3.detected);

    Overlord ova4(do_auc, nrows, ngroups);
    auto state4 = runner.run(sparse_column.get(), groups.data(), ngroups, ova4);
    compare_almost_equal(state.means, state4.means);
    compare_almost_equal(state.variances, state4.variances);
    EXPECT_EQ(state.detected, state4.detected);

    // Checking AUCs.
    EXPECT_EQ(ova.store, ova2.store);
    EXPECT_EQ(ova.store, ova3.store);
    EXPECT_EQ(ova.store, ova4.store);
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

