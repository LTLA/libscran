#include <gtest/gtest.h>

#include "../data/data.h"
#include "../utils/compare_vectors.h"

#include "tatami/base/DenseMatrix.hpp"
#include "scran/quality_control/PerCellAdtQcFilters.hpp"

#include <cmath>
#include <numeric>

class PerCellAdtQcFiltersTester : public ::testing::Test {
protected:
    void SetUp() {
        mat = std::unique_ptr<tatami::NumericMatrix>(new tatami::DenseRowMatrix<double>(sparse_nrow, sparse_ncol, sparse_matrix));
    }
protected:
    std::shared_ptr<tatami::NumericMatrix> mat;
    scran::PerCellAdtQcMetrics qc;

    std::vector<int> keep_i = { 0, 1, 2, 3, 11, 12, 13, 14 };
};

TEST_F(PerCellAdtQcFiltersTester, NoSubset) {
    auto qcres = qc.run(mat.get(), {});

    scran::PerCellAdtQcFilters filters;
    auto res = filters.set_nmads(0.5).run(qcres);

    // something got filtered, right?
    EXPECT_TRUE(std::accumulate(res.filter_by_detected.begin(), res.filter_by_detected.end(), 0) > 0); 

    scran::IsOutlier ref;
    ref.set_nmads(0.5).set_lower(true).set_upper(false).set_log(true);
    auto refres = ref.run(qcres.detected);
    EXPECT_EQ(refres.outliers, res.filter_by_detected);
    EXPECT_EQ(refres.thresholds.lower, res.thresholds.detected);
}

TEST_F(PerCellAdtQcFiltersTester, OneSubset) {
    std::vector<uint8_t> keep_s(mat->nrow());
    for (auto i : keep_i) { keep_s[i] = 1; }
    auto qcres = qc.run(mat.get(), std::vector<const uint8_t*>(1, keep_s.data()));

    scran::PerCellAdtQcFilters filters;
    auto res = filters.set_nmads(1).run(qcres);
    EXPECT_TRUE(std::accumulate(res.filter_by_subset_totals[0].begin(), res.filter_by_subset_totals[0].end(), 0) > 0); 

    scran::IsOutlier ref;
    auto refres = ref.set_nmads(1).set_lower(false).set_log(true).run(qcres.subset_totals[0]);
    EXPECT_EQ(refres.outliers, res.filter_by_subset_totals[0]);
    EXPECT_EQ(refres.thresholds.upper, res.thresholds.subset_totals[0]);
}

TEST_F(PerCellAdtQcFiltersTester, Blocking) {
    std::vector<uint8_t> keep_s(mat->nrow());
    for (auto i : keep_i) { keep_s[i] = 1; }
    auto qcres = qc.run(mat.get(), std::vector<const uint8_t*>(1, keep_s.data()));

    std::vector<int> block(mat->ncol());
    for (size_t i = 0; i < block.size(); ++i) { block[i] = i % 5; }
    scran::PerCellAdtQcFilters filters;
    auto res = filters.set_nmads(1).run_blocked(qcres, block.data());

    scran::IsOutlier ref;
    auto refres = ref.set_nmads(1).set_lower(true).set_upper(false).set_log(true).run_blocked(qcres.detected, block.data());
    EXPECT_EQ(refres.outliers, res.filter_by_detected);
    EXPECT_EQ(refres.thresholds.lower, res.thresholds.detected);
}

TEST_F(PerCellAdtQcFiltersTester, Overall) {
    std::vector<uint8_t> keep_s(mat->nrow());
    for (auto i : keep_i) { keep_s[i] = 1; }
    auto qcres = qc.run(mat.get(), std::vector<const uint8_t*>(1, keep_s.data()));

    scran::PerCellAdtQcFilters filters;
    auto res = filters.run(qcres);

    std::vector<uint8_t> vec(mat->ncol());
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = res.filter_by_detected[i] || res.filter_by_subset_totals[0][i];
    }
    EXPECT_EQ(res.overall_filter, vec);
}
