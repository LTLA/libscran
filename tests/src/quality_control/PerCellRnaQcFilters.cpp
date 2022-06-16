#include <gtest/gtest.h>

#include "../data/data.h"
#include "../utils/compare_vectors.h"

#include "tatami/base/DenseMatrix.hpp"
#include "scran/quality_control/PerCellRnaQcMetrics.hpp"
#include "scran/quality_control/PerCellRnaQcFilters.hpp"

#include <cmath>
#include <numeric>

class PerCellRnaQcFiltersTester : public ::testing::Test {
protected:
    void SetUp() {
        mat = std::unique_ptr<tatami::NumericMatrix>(new tatami::DenseRowMatrix<double>(sparse_nrow, sparse_ncol, sparse_matrix));
    }

    template<class V>
    static scran::IsOutlier::Results<uint8_t> run(const scran::IsOutlier& ref, const V& vec) {
        return ref.run(vec.size(), vec.data());
    }

    template<class V, typename B>
    static scran::IsOutlier::Results<uint8_t> run_blocked(const scran::IsOutlier& ref, const V& vec, B block) {
        return ref.run_blocked(vec.size(), block, vec.data());
    }
protected:
    std::shared_ptr<tatami::NumericMatrix> mat;
    scran::PerCellRnaQcMetrics qc;
};

TEST_F(PerCellRnaQcFiltersTester, NoSubset) {
    auto qcres = qc.run(mat.get(), {});

    // Default NMADs works correctly.
    scran::PerCellRnaQcFilters filters;
    auto res = filters.run(qcres);

    scran::IsOutlier ref;
    ref.set_lower(true).set_upper(false).set_log(true);

    {
        auto refres = run(ref, qcres.sums);
        EXPECT_EQ(refres.outliers, res.filter_by_sums);
        EXPECT_EQ(refres.thresholds.lower, res.thresholds.sums);
    }

    {
        auto refres = run(ref, qcres.detected);
        EXPECT_EQ(refres.outliers, res.filter_by_detected);
        EXPECT_EQ(refres.thresholds.lower, res.thresholds.detected);
    }

    // Lowering the nmads to get some actual filtering happening.
    auto res2 = filters.set_nmads(0.5).run(qcres);
    ref.set_nmads(0.5);

    {
        auto refres = run(ref, qcres.sums);
        EXPECT_TRUE(std::accumulate(refres.outliers.begin(), refres.outliers.end(), 0) > 0);
        EXPECT_EQ(refres.outliers, res2.filter_by_sums);
        EXPECT_EQ(refres.thresholds.lower, res2.thresholds.sums);
    }

    {
        auto refres = run(ref, qcres.detected);
        EXPECT_TRUE(std::accumulate(refres.outliers.begin(), refres.outliers.end(), 0) > 0);
        EXPECT_EQ(refres.outliers, res2.filter_by_detected);
        EXPECT_EQ(refres.thresholds.lower, res2.thresholds.detected);
    }
}

std::vector<int> keep_i = { 0, 1, 2, 3, 11, 12, 13, 14 };

TEST_F(PerCellRnaQcFiltersTester, OneSubset) {
    std::vector<uint8_t> keep_s(mat->nrow());
    for (auto i : keep_i) { keep_s[i] = 1; }
    auto qcres = qc.run(mat.get(), std::vector<const uint8_t*>(1, keep_s.data()));

    scran::PerCellRnaQcFilters filters;
    auto res = filters.set_nmads(1).run(qcres);

    scran::IsOutlier ref;
    ref.set_nmads(1).set_lower(false).set_upper(true).set_log(false);

    auto refres = run(ref, qcres.subset_proportions[0]);
    EXPECT_TRUE(std::accumulate(refres.outliers.begin(), refres.outliers.end(), 0) > 0);

    EXPECT_EQ(refres.outliers, res.filter_by_subset_proportions[0]);
    EXPECT_EQ(refres.thresholds.upper, res.thresholds.subset_proportions[0]);
}

TEST_F(PerCellRnaQcFiltersTester, Blocking) {
    std::vector<uint8_t> keep_s(mat->nrow());
    for (auto i : keep_i) { keep_s[i] = 1; }
    auto qcres = qc.run(mat.get(), std::vector<const uint8_t*>(1, keep_s.data()));

    std::vector<int> block(mat->ncol());
    for (size_t i = 0; i < block.size(); ++i) { block[i] = i % 5; }
    scran::PerCellRnaQcFilters filters;
    auto res = filters.set_nmads(1).run_blocked(qcres, block.data());

    scran::IsOutlier ref;
    ref.set_nmads(1).set_lower(true).set_upper(false).set_log(true);

    {
        auto refres = run_blocked(ref, qcres.sums, block.data());
        EXPECT_EQ(refres.outliers, res.filter_by_sums);
        EXPECT_EQ(refres.thresholds.lower, res.thresholds.sums);
    }

    {
        auto refres = run_blocked(ref, qcres.detected, block.data());
        EXPECT_EQ(refres.outliers, res.filter_by_detected);
        EXPECT_EQ(refres.thresholds.lower, res.thresholds.detected);
    }

    {
        ref.set_lower(false).set_upper(true).set_log(false);
        auto refres = run_blocked(ref, qcres.subset_proportions[0], block.data());
        EXPECT_EQ(refres.outliers, res.filter_by_subset_proportions[0]);
        EXPECT_EQ(refres.thresholds.upper, res.thresholds.subset_proportions[0]);
    }
}

TEST_F(PerCellRnaQcFiltersTester, Overall) {
    std::vector<uint8_t> keep_s(mat->nrow());
    for (auto i : keep_i) { keep_s[i] = 1; }
    auto qcres = qc.run(mat.get(), std::vector<const uint8_t*>(1, keep_s.data()));

    scran::PerCellRnaQcFilters filters;
    auto res = filters.run(qcres);

    std::vector<uint8_t> vec(mat->ncol());
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = res.filter_by_sums[i] || res.filter_by_detected[i] || res.filter_by_subset_proportions[0][i];
    }
    EXPECT_EQ(res.overall_filter, vec);
}
