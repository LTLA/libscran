#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "../data/Simulator.hpp"
#include "utils.h"

#include "scran/quality_control/SuggestCrisprQcFilters.hpp"

#include <cmath>
#include <numeric>

class SuggestCrisprQcFiltersTest : public ::testing::Test {
protected:
    std::shared_ptr<tatami::NumericMatrix> mat;

    void SetUp() {
        size_t nr = 15, nc = 1001; // very few rows to simulate high total counts.
        Simulator sim;
        sim.lower = 0;
        sim.upper = 1000;
        auto mat0 = sim.matrix(nr, nc);
        mat.reset(new decltype(mat0)(std::move(mat0)));
    }
};

TEST_F(SuggestCrisprQcFiltersTest, BasicCheck) {
    auto qcres = scran::PerCellCrisprQcMetrics().run(mat.get());

    scran::SuggestCrisprQcFilters filters;
    auto res = filters.run(qcres);
    EXPECT_EQ(res.max_count.size(), 1);
    EXPECT_TRUE(res.max_count[0] > 0);

    // Just checking that we're operating on the right scale here.
    filters.set_num_mads(0);
    auto res2 = filters.run(qcres);
    EXPECT_TRUE(res2.max_count[0] > res.max_count[0]);
    EXPECT_TRUE(res2.max_count[0] > 100);
}

TEST_F(SuggestCrisprQcFiltersTest, EdgeCase) {
    // Everyone has a 100% proportions.
    size_t NR = 5, NC = 100;
    std::vector<double> buffer(NR * NC);
    std::vector<double> totals(NC);
    std::mt19937_64 rng(123456);

    for (size_t c = 0; c < NC; ++c) {
        double y = 100 + rng() % 2000;
        buffer[c * NR + rng() % NR] = y;
        totals[c] = y;
    }

    tatami::DenseColumnMatrix<double, int> mat(NR, NC, std::move(buffer));
    auto qcres = scran::PerCellCrisprQcMetrics().run(&mat);
    auto filt = scran::SuggestCrisprQcFilters().run(qcres);
    
    auto ref = scran::ComputeMedianMad().set_log(true).run(NC, totals.data());
    EXPECT_DOUBLE_EQ(filt.max_count[0], std::exp(ref.medians[0] - ref.mads[0] * 3));
}

TEST_F(SuggestCrisprQcFiltersTest, EdgeCaseZeros) {
    // All-zero cells do not participate at any step; NA proportions are
    // implicitly removed during the median calculation and the subsetting to
    // define the thresholds, and the NA sums are removed during the filtering.
    size_t NR = 5, NC = 105;
    std::vector<double> buffer(NR * NC);
    std::mt19937_64 rng(1234567);
    size_t nempty = 4;

    for (size_t c = nempty; c < NC; ++c) {
        double y = 100 + rng() % 2000;
        buffer[c * NR + rng() % NR] = y;
    }

    tatami::DenseColumnMatrix<double, int> mat(NR, NC, buffer);
    auto qcres = scran::PerCellCrisprQcMetrics().run(&mat);
    auto filt = scran::SuggestCrisprQcFilters().run(qcres);
    auto fres = filt.filter(qcres);

    tatami::DenseColumnMatrix<double, int> mat2(NR, NC - nempty, std::vector<double>(buffer.begin() + NR * nempty, buffer.end()));
    auto qcres2 = scran::PerCellCrisprQcMetrics().run(&mat2);
    auto filt2 = scran::SuggestCrisprQcFilters().run(qcres2);
    auto fres2 = filt2.filter(qcres2);

    EXPECT_EQ(filt.max_count, filt2.max_count);
    EXPECT_EQ(std::vector<uint8_t>(fres.begin() + nempty, fres.end()), fres2);
    EXPECT_EQ(std::vector<uint8_t>(fres.begin(), fres.begin() + nempty), std::vector<uint8_t>(nempty));
}

TEST_F(SuggestCrisprQcFiltersTest, Blocking) {
    auto qcres = scran::PerCellCrisprQcMetrics().run(mat.get());

    size_t nblocks = 3;
    std::vector<int> block = quality_control::create_blocks(mat->ncol(), nblocks);

    scran::SuggestCrisprQcFilters filters;
    auto filt = filters.run_blocked(qcres, block.data());
    EXPECT_EQ(filt.max_count.size(), nblocks);

    // Reference calculations.
    for (size_t b = 0; b < nblocks; ++b) {
        std::vector<int> keep;
        for (size_t c = 0; c < mat->ncol(); ++c) {
            if (block[c] == static_cast<int>(b)) {
                keep.push_back(c);
            }
        }

        auto sub = tatami::make_DelayedSubset<1>(mat, std::move(keep));
        auto submet = scran::PerCellCrisprQcMetrics().run(sub.get());
        auto subfilt = scran::SuggestCrisprQcFilters().run(submet);

        EXPECT_EQ(filt.max_count[b], subfilt.max_count[0]);       
    }
}

TEST_F(SuggestCrisprQcFiltersTest, Filters) {
    auto qcres = scran::PerCellCrisprQcMetrics().run(mat.get());
    scran::SuggestCrisprQcFilters filters;

    // Single block.
    {
        auto thresh = filters.run(qcres);
        auto discards = thresh.filter(qcres);

        std::vector<uint8_t> vec(mat->ncol());
        size_t discarded = 0;
        for (size_t i = 0; i < vec.size(); ++i) {
            vec[i] = qcres.max_proportion[i] * qcres.sums[i] < thresh.max_count[0];
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
            vec[i] = qcres.max_proportion[i] * qcres.sums[i] < thresh.max_count[b];
            discarded += vec[i];
        }
        EXPECT_EQ(discards, vec);
        EXPECT_TRUE(discarded > 0 && discarded < vec.size()); // some filtered, some not.
    }
}
