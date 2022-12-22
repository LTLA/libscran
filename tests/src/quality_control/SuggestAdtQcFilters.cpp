#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "../data/Simulator.hpp"
#include "utils.h"

#include "scran/quality_control/SuggestAdtQcFilters.hpp"

#include <cmath>
#include <numeric>

class SuggestAdtQcFiltersTest : public ::testing::Test {
protected:
    std::shared_ptr<tatami::NumericMatrix> mat;

    void SetUp() {
        size_t nr = 57, nc = 213;
        Simulator sim;
        sim.lower = 0;
        auto mat0 = sim.matrix(nr, nc);
        mat.reset(new decltype(mat0)(std::move(mat0)));
    }
};

TEST_F(SuggestAdtQcFiltersTest, NoSubset) {
    auto qcres = scran::PerCellAdtQcMetrics().run(mat.get(), {});

    scran::SuggestAdtQcFilters filters;
    filters.set_min_detected_drop(0);
    auto res = filters.run(qcres);
    EXPECT_EQ(res.detected.size(), 1);

    // Reference comparison.
    scran::ComputeMedianMad meddler;
    meddler.set_log(true);
    auto comp = meddler.run(qcres.detected.size(), qcres.detected.data());
    EXPECT_DOUBLE_EQ(res.detected[0], std::exp(comp.medians[0] - 3 * comp.mads[0]));
}

TEST_F(SuggestAdtQcFiltersTest, MinDetectedDrop) {
    auto qcres = scran::PerCellAdtQcMetrics().run(mat.get(), {});

    scran::SuggestAdtQcFilters filters;
    filters.set_min_detected_drop(0.9);
    auto res = filters.run(qcres);
    EXPECT_EQ(res.detected.size(), 1);

    // Reference comparison.
    scran::ComputeMedianMad meddler;
    meddler.set_log(true);
    auto comp = meddler.run(qcres.detected.size(), qcres.detected.data());
    EXPECT_DOUBLE_EQ(res.detected[0], std::exp(comp.medians[0]) * 0.1);
}

TEST_F(SuggestAdtQcFiltersTest, OneSubset) {
    std::vector<size_t> keep_i = { 0, 5, 7, 8, 9, 10, 16, 17 };
    auto keep_s = quality_control::to_filter(mat->nrow(), keep_i);
    auto qcres = scran::PerCellAdtQcMetrics().run(mat.get(), std::vector<const int*>(1, keep_s.data()));

    scran::SuggestAdtQcFilters filters;
    auto res = filters.run(qcres);
    EXPECT_EQ(res.subset_totals.size(), 1);
    EXPECT_EQ(res.subset_totals[0].size(), 1);

    // Reference comparison.
    scran::ComputeMedianMad meddler;
    meddler.set_log(true);
    auto comp = meddler.run(qcres.subset_totals[0].size(), qcres.subset_totals[0].data());
    EXPECT_DOUBLE_EQ(res.subset_totals[0][0], std::exp(comp.medians[0] + 3 * comp.mads[0]));
}

TEST_F(SuggestAdtQcFiltersTest, Blocking) {
    std::vector<size_t> keep_i = { 5, 7, 9, 11, 13, 15, 21 };
    auto keep_s = quality_control::to_filter(mat->nrow(), keep_i);
    auto qcres = scran::PerCellAdtQcMetrics().run(mat.get(), std::vector<const int*>(1, keep_s.data()));

    size_t nblocks = 3;
    std::vector<int> block = quality_control::create_blocks(mat->ncol(), nblocks);

    scran::SuggestAdtQcFilters filters;
    auto res = filters.run_blocked(qcres, block.data());
    EXPECT_EQ(res.detected.size(), nblocks);
    EXPECT_EQ(res.subset_totals.size(), 1);
    EXPECT_EQ(res.subset_totals[0].size(), nblocks);

    // Reference calculation.
    scran::ComputeMedianMad meddler;
    meddler.set_log(true);
    auto comp = meddler.run_blocked(qcres.subset_totals[0].size(), block.data(), qcres.subset_totals[0].data());
    for (size_t n = 0; n < nblocks; ++n) {
        EXPECT_DOUBLE_EQ(res.subset_totals[0][n], std::exp(comp.medians[n] + 3 * comp.mads[n]));
    }
}

TEST_F(SuggestAdtQcFiltersTest, Filters) {
    std::vector<size_t> keep_i = { 1, 3, 5, 10, 23, 45, 50, 51 };
    auto keep_s = quality_control::to_filter(mat->nrow(), keep_i);
    auto qcres = scran::PerCellAdtQcMetrics().run(mat.get(), std::vector<const int*>(1, keep_s.data()));

    scran::SuggestAdtQcFilters filters;
    filters.set_detected_num_mads(0.5);
    filters.set_subset_num_mads(0.5);

    // Single block.
    {
        auto thresh = filters.run(qcres);
        auto discards = thresh.filter(qcres);

        std::vector<uint8_t> vec(mat->ncol());
        size_t discarded = 0;
        for (size_t i = 0; i < vec.size(); ++i) {
            vec[i] = qcres.detected[i] < thresh.detected[0] || qcres.subset_totals[0][i] > thresh.subset_totals[0][0];
            discarded += vec[i];
        }
        EXPECT_EQ(discards, vec);
        EXPECT_TRUE(discarded > 0 && discarded < vec.size()); // some filtered, some not.

        auto discards2 = thresh.filter_blocked(qcres, static_cast<int*>(NULL));
        EXPECT_EQ(discards, discards2);
    }

    // Multiple blocks.
    {
        size_t nblocks = 3;
        std::vector<int> block = quality_control::create_blocks(mat->ncol(), nblocks);

        auto thresh = filters.run_blocked(qcres, block.data());
        EXPECT_ANY_THROW(thresh.filter(qcres));
        auto discards = thresh.filter_blocked(qcres, block.data());

        std::vector<uint8_t> vec(mat->ncol());
        size_t discarded = 0;
        for (size_t i = 0; i < vec.size(); ++i) {
            auto b = block[i];
            vec[i] = qcres.detected[i] < thresh.detected[b] || qcres.subset_totals[0][i] > thresh.subset_totals[0][b];
            discarded += vec[i];
        }
        EXPECT_EQ(discards, vec);
        EXPECT_TRUE(discarded > 0 && discarded < vec.size()); // some filtered, some not.
    }
}
