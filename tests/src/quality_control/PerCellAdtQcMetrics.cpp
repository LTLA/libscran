#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "../data/Simulator.hpp"
#include "utils.h"

#include "tatami/tatami.hpp"

#include "scran/quality_control/PerCellAdtQcMetrics.hpp"

class PerCellAdtQcMetricsTest : public ::testing::Test {
protected:
    std::shared_ptr<tatami::NumericMatrix> mat;

    void SetUp() {
        size_t nr = 45, nc = 50;
        Simulator sim;
        sim.lower = 0;
        auto mat0 = sim.matrix(nr, nc);
        mat.reset(new decltype(mat0)(std::move(mat0)));
    }
};

TEST_F(PerCellAdtQcMetricsTest, NoSubset) {
    scran::PerCellAdtQcMetrics qcfun;
    auto res = qcfun.run(mat.get(), {});
    EXPECT_EQ(res.sums, tatami::column_sums(mat.get()));
    EXPECT_EQ(res.detected, quality_control::compute_num_detected(mat.get()));
    EXPECT_TRUE(res.subset_totals.empty());
}

TEST_F(PerCellAdtQcMetricsTest, OneSubset) {
    std::vector<size_t> keep_i = { 0, 5, 7, 8, 9, 10, 16, 17 };
    auto keep_s = quality_control::to_filter(mat->nrow(), keep_i);
    std::vector<const int*> subs(1, keep_s.data());

    scran::PerCellAdtQcMetrics qcfun;
    qcfun.set_num_threads(2); // just for some coverage.
    auto res = qcfun.run(mat.get(), subs);

    auto submat = tatami::make_DelayedSubset<0>(mat, keep_i);
    auto subsums = tatami::column_sums(submat.get());

    EXPECT_EQ(res.sums, tatami::column_sums(mat.get()));
    EXPECT_EQ(res.detected, quality_control::compute_num_detected(mat.get()));
    EXPECT_EQ(res.subset_totals[0], subsums);
}
