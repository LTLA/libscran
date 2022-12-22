#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "../data/Simulator.hpp"
#include "utils.h"

#include "scran/quality_control/PerCellRnaQcMetrics.hpp"
#include "scran/quality_control/SuggestRnaQcFilters.hpp"

#include <cmath>
#include <numeric>

class SuggestRnaQcFiltersTest : public ::testing::Test {
protected:
    std::shared_ptr<tatami::NumericMatrix> mat;

    void SetUp() {
        size_t nr = 555, nc = 313;
        Simulator sim;
        sim.lower = 0;
        auto mat0 = sim.matrix(nr, nc);
        mat.reset(new decltype(mat0)(std::move(mat0)));
    }
};

TEST_F(SuggestRnaQcFiltersTest, NoSubset) {
    auto qcres = scran::PerCellRnaQcMetrics().run(mat.get(), {});

    scran::SuggestRnaQcFilters filters;
    auto thresh = filters.run(qcres);

    scran::ComputeMedianMad ref;
    ref.set_log(true);

    auto sstats = ref.run(qcres.sums.size(), qcres.sums.data());
    EXPECT_DOUBLE_EQ(thresh.sums[0], std::exp(sstats.medians[0] - sstats.mads[0] * 3));

    auto dstats = ref.run(qcres.detected.size(), qcres.detected.data());
    EXPECT_DOUBLE_EQ(thresh.detected[0], std::exp(dstats.medians[0] - dstats.mads[0] * 3));
}

TEST_F(SuggestRnaQcFiltersTest, OneSubset) {
    std::vector<size_t> keep_i { 0, 1, 2, 3, 11, 12, 13, 14 };
    auto keep_s = quality_control::to_filter(mat->nrow(), keep_i);
    auto qcres = scran::PerCellRnaQcMetrics().run(mat.get(), std::vector<const int*>(1, keep_s.data()));

    scran::SuggestRnaQcFilters filters;
    auto thresh = filters.run(qcres);

    scran::ComputeMedianMad ref;
    auto substats = ref.run(qcres.subset_proportions[0].size(), qcres.subset_proportions[0].data());
    EXPECT_EQ(thresh.subset_proportions[0][0], substats.medians[0] + substats.mads[0] * 3);
}

TEST_F(SuggestRnaQcFiltersTest, Blocking) {
    std::vector<size_t> keep_i { 2, 4, 8, 16, 32, 64, 128, 256 };
    auto keep_s = quality_control::to_filter(mat->nrow(), keep_i);
    auto qcres = scran::PerCellRnaQcMetrics().run(mat.get(), std::vector<const int*>(1, keep_s.data()));

    size_t nblocks = 5;
    auto block = quality_control::create_blocks(mat->ncol(), nblocks);

    scran::SuggestRnaQcFilters filters;
    auto thresh = filters.run_blocked(qcres, block.data());
    EXPECT_EQ(thresh.detected.size(), nblocks);
    EXPECT_EQ(thresh.sums.size(), nblocks);
    EXPECT_EQ(thresh.subset_proportions.size(), 1);
    EXPECT_EQ(thresh.subset_proportions[0].size(), nblocks);

    // Compare to a reference calculation.
    scran::ComputeMedianMad ref;
    ref.set_log(true);
    auto refres = ref.run_blocked(qcres.sums.size(), block.data(), qcres.sums.data());
    for (size_t b = 0; b < nblocks; ++b) {
        EXPECT_DOUBLE_EQ(thresh.sums[b], std::exp(refres.medians[b] - 3 * refres.mads[b]));
    }
}

TEST_F(SuggestRnaQcFiltersTest, Filters) {
    std::vector<size_t> keep_i = { 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377 };
    auto keep_s = quality_control::to_filter(mat->nrow(), keep_i);
    auto qcres = scran::PerCellRnaQcMetrics().run(mat.get(), std::vector<const int*>(1, keep_s.data()));

    scran::SuggestRnaQcFilters filters;
    filters.set_detected_num_mads(0.5);
    filters.set_subset_num_mads(0.5);

    // Single block.
    {
        auto thresh = filters.run(qcres);
        auto discards = thresh.filter(qcres);

        std::vector<uint8_t> vec(mat->ncol());
        size_t discarded = 0;
        for (size_t i = 0; i < vec.size(); ++i) {
            vec[i] = qcres.detected[i] < thresh.detected[0] 
                || qcres.sums[i] < thresh.sums[0] 
                || qcres.subset_proportions[0][i] > thresh.subset_proportions[0][0];
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
            vec[i] = qcres.detected[i] < thresh.detected[b] 
                || qcres.sums[i] < thresh.sums[b] 
                || qcres.subset_proportions[0][i] > thresh.subset_proportions[0][b];
            discarded += vec[i];
        }
        EXPECT_EQ(discards, vec);
        EXPECT_TRUE(discarded > 0 && discarded < vec.size()); // some filtered, some not.
    }
}
