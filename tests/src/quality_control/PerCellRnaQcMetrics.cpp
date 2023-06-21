#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "../data/Simulator.hpp"
#include "utils.h"

#include "tatami/tatami.hpp"

#include "scran/quality_control/PerCellRnaQcMetrics.hpp"

class PerCellRnaQcMetricsTest : public ::testing::Test {
protected:
    std::shared_ptr<tatami::NumericMatrix> mat;

    void SetUp() {
        size_t nr = 21, nc = 99;
        Simulator sim;
        sim.lower = 0;
        auto mat0 = sim.matrix(nr, nc);
        mat.reset(new decltype(mat0)(std::move(mat0)));
    }
};

TEST_F(PerCellRnaQcMetricsTest, NoSubset) {
    scran::PerCellRnaQcMetrics qcfun;
    auto res = qcfun.run(mat.get(), {});
    EXPECT_EQ(res.sums, tatami::column_sums(mat.get()));
    EXPECT_EQ(res.detected, quality_control::compute_num_detected(mat.get()));
    EXPECT_TRUE(res.subset_proportions.empty());
}

TEST_F(PerCellRnaQcMetricsTest, OneSubset) {
    std::vector<size_t> keep_i = { 0, 5, 7, 8, 9, 10, 16, 17 };
    auto keep_s = quality_control::to_filter(mat->nrow(), keep_i);
    std::vector<const int*> subs(1, keep_s.data());

    scran::PerCellRnaQcMetrics qcfun;
    auto res = qcfun.run(mat.get(), subs);

    auto submat = tatami::make_DelayedSubset<0>(mat, keep_i);
    auto subsums = tatami::column_sums(submat.get());
    auto it = res.sums.begin();
    for (auto& s : subsums) {
        if (*it) {
            s /= *it;
        } else {
            // Can't be bothered to do special handling for NaNs here.
            s = -100;
        }
        ++it;
    }

    for (size_t i = 0; i < subsums.size(); ++i) {
        auto& x = res.subset_proportions[0][i];
        if (std::isnan(x)) {
            x = -100;
        }
    }

    EXPECT_EQ(res.sums, tatami::column_sums(mat.get()));
    EXPECT_EQ(res.detected, quality_control::compute_num_detected(mat.get()));
    EXPECT_EQ(res.subset_proportions[0], subsums);
}

TEST_F(PerCellRnaQcMetricsTest, NASubsets) {
    std::vector<size_t> keep_i = { 0, 5, 7, 8, 9, 10, 16, 17 };
    auto keep_s = quality_control::to_filter(mat->nrow(), keep_i);

    std::vector<double> nothing(100);
    auto dense_zero = std::unique_ptr<tatami::NumericMatrix>(new tatami::DenseColumnMatrix<double>(20, 5, std::move(nothing)));
    std::vector<const int*> subs = { keep_s.data() };

    scran::PerCellRnaQcMetrics qcfun;
    qcfun.set_num_threads(2); // just for some coverage.
    auto res = qcfun.run(dense_zero.get(), subs);

    EXPECT_EQ(res.sums, std::vector<double>(dense_zero->ncol()));
    EXPECT_EQ(res.detected, std::vector<int>(dense_zero->ncol()));
    EXPECT_TRUE(std::isnan(res.subset_proportions[0][0]));
    EXPECT_TRUE(std::isnan(res.subset_proportions[0][dense_zero->ncol()-1]));
}
