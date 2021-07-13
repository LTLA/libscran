#include <gtest/gtest.h>

#include "../data/data.h"
#include "../utils/compare_vectors.h"

#include "tatami/base/DenseMatrix.hpp"
#include "scran/quality_control/PerCellQCMetrics.hpp"
#include "scran/quality_control/PerCellQCFilters.hpp"

#include <cmath>

class PerCellQCFiltersTester : public ::testing::Test {
protected:
    void SetUp() {
        mat = std::unique_ptr<tatami::NumericMatrix>(new tatami::DenseRowMatrix<double>(sparse_nrow, sparse_ncol, sparse_matrix));
    }
protected:
    std::shared_ptr<tatami::NumericMatrix> mat;
    scran::PerCellQCMetrics qc;
};

TEST_F(PerCellQCFiltersTester, NoSubset) {
    auto qcres = qc.run(mat.get(), {});

    scran::PerCellQCFilters filters;
    auto res = filters.run(qcres, nullptr);

    scran::IsOutlier ref;
    auto refres = ref.set_lower(true).set_upper(false).set_log(true).run(qcres.sums, nullptr);
    EXPECT_EQ(refres.outliers, res.filter_by_sums);
    EXPECT_EQ(refres.thresholds.lower, res.thresholds.sums);

    refres = ref.run(qcres.detected, nullptr);
    EXPECT_EQ(refres.outliers, res.filter_by_detected);
    EXPECT_EQ(refres.thresholds.lower, res.thresholds.detected);

    // Lowering the nmads to get some actual filtering happening.
    auto res2 = filters.set_nmads(0.5).run(qcres, nullptr);

    refres = ref.set_nmads(0.5).run(qcres.sums, nullptr);
    EXPECT_EQ(refres.outliers, res2.filter_by_sums);
    EXPECT_EQ(refres.thresholds.lower, res2.thresholds.sums);

    refres = ref.run(qcres.detected, nullptr);
    EXPECT_EQ(refres.outliers, res2.filter_by_detected);
    EXPECT_EQ(refres.thresholds.lower, res2.thresholds.detected);
}

std::vector<int> keep_i = { 0, 1, 2, 3, 11, 12, 13, 14 };

TEST_F(PerCellQCFiltersTester, OneSubset) {
    std::vector<uint8_t> keep_s(mat->nrow());
    for (auto i : keep_i) { keep_s[i] = 1; }
    auto qcres = qc.run(mat.get(), std::vector<const uint8_t*>(1, keep_s.data()));

    scran::PerCellQCFilters filters;
    auto res = filters.set_nmads(1).run(qcres, nullptr);

    scran::IsOutlier ref;
    auto refres = ref.set_nmads(1).set_lower(false).run(qcres.subset_proportions[0], nullptr);
    EXPECT_EQ(refres.outliers, res.filter_by_subset_proportions[0]);
    EXPECT_EQ(refres.thresholds.upper, res.thresholds.subset_proportions[0]);
}

TEST_F(PerCellQCFiltersTester, Blocking) {
    std::vector<uint8_t> keep_s(mat->nrow());
    for (auto i : keep_i) { keep_s[i] = 1; }
    auto qcres = qc.run(mat.get(), std::vector<const uint8_t*>(1, keep_s.data()));

    std::vector<int> block(mat->ncol());
    for (size_t i = 0; i < block.size(); ++i) { block[i] = i % 5; }
    scran::PerCellQCFilters filters;
    auto res = filters.set_nmads(1).run(qcres, block.data());

    scran::IsOutlier ref;
    auto refres = ref.set_nmads(1).set_lower(true).set_upper(false).set_log(true).run(qcres.sums, block.data());
    EXPECT_EQ(refres.outliers, res.filter_by_sums);
    EXPECT_EQ(refres.thresholds.lower, res.thresholds.sums);

    refres = ref.run(qcres.detected, block.data());
    EXPECT_EQ(refres.outliers, res.filter_by_detected);
    EXPECT_EQ(refres.thresholds.lower, res.thresholds.detected);

    refres = ref.set_lower(false).set_upper(true).set_log(false).run(qcres.subset_proportions[0], block.data());
    EXPECT_EQ(refres.outliers, res.filter_by_subset_proportions[0]);
    EXPECT_EQ(refres.thresholds.upper, res.thresholds.subset_proportions[0]);
}

TEST_F(PerCellQCFiltersTester, Overall) {
    std::vector<uint8_t> keep_s(mat->nrow());
    for (auto i : keep_i) { keep_s[i] = 1; }
    auto qcres = qc.run(mat.get(), std::vector<const uint8_t*>(1, keep_s.data()));

    scran::PerCellQCFilters filters;
    auto res = filters.run(qcres, nullptr);

    std::vector<uint8_t> vec(mat->ncol());
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = res.filter_by_sums[i] || res.filter_by_detected[i] || res.filter_by_subset_proportions[0][i];
    }
    EXPECT_EQ(res.overall_filter, vec);
}
