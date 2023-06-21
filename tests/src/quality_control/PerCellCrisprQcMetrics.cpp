#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "../data/Simulator.hpp"
#include "utils.h"

#include "tatami/tatami.hpp"

#include "scran/quality_control/PerCellCrisprQcMetrics.hpp"

class PerCellCrisprQcMetricsTest : public ::testing::Test {
protected:
    std::shared_ptr<tatami::NumericMatrix> mat;

    void SetUp() {
        size_t nr = 99, nc = 777;
        Simulator sim;
        sim.lower = 0;
        auto mat0 = sim.matrix(nr, nc);
        mat.reset(new decltype(mat0)(std::move(mat0)));
    }
};

TEST_F(PerCellCrisprQcMetricsTest, Basic) {
    scran::PerCellCrisprQcMetrics qcfun;
    auto res = qcfun.run(mat.get());
    EXPECT_EQ(res.sums, tatami::column_sums(mat.get()));
    EXPECT_EQ(res.detected, quality_control::compute_num_detected(mat.get()));
    EXPECT_EQ(res.max_proportion.size(), mat->ncol());
    EXPECT_EQ(res.max_index.size(), mat->ncol());
}

TEST_F(PerCellCrisprQcMetricsTest, NAMaxed) {
    std::vector<double> nothing(100);
    auto dense_zero = std::unique_ptr<tatami::NumericMatrix>(new tatami::DenseColumnMatrix<double>(20, 5, std::move(nothing)));

    scran::PerCellCrisprQcMetrics qcfun;
    qcfun.set_num_threads(2); // just for some coverage.
    auto res = qcfun.run(dense_zero.get());

    EXPECT_EQ(res.sums, std::vector<double>(dense_zero->ncol()));
    EXPECT_EQ(res.detected, std::vector<int>(dense_zero->ncol()));
    EXPECT_TRUE(std::isnan(res.max_proportion[0]));
    EXPECT_TRUE(std::isnan(res.max_proportion[dense_zero->ncol()-1]));
}
