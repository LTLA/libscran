#include <gtest/gtest.h>

#include "../data/data.h"
#include "../utils/compare_vectors.h"

#include "tatami/base/DenseMatrix.hpp"
#include "scran/qc/PerCellQCMetrics.hpp"
#include "scran/qc/PerCellQCFilters.hpp"

#include <cmath>

class PerCellQCFiltersTester : public ::testing::Test {
protected:
    void SetUp() {
        mat = std::unique_ptr<tatami::numeric_matrix>(new tatami::DenseRowMatrix<double>(sparse_nrow, sparse_ncol, sparse_matrix));
    }
protected:
    std::shared_ptr<tatami::numeric_matrix> mat;
    scran::PerCellQCMetrics<> qc;
};

TEST_F(PerCellQCFiltersTester, NoSubset) {
    auto qcres = qc.run(mat.get());

    scran::PerCellQCFilters filters;
    auto res = filters.run(qcres);

    scran::IsOutlier<> ref;
    auto refres = ref.set_lower(true).set_upper(false).set_log(true).run(mat->ncol(), qcres.sums);
    compare_vectors(mat->ncol(), refres.outliers, res.filter_by_sums);
    EXPECT_EQ(refres.lower_thresholds, res.sums_thresholds);

    refres = ref.run(mat->ncol(), qcres.detected);
    compare_vectors(mat->ncol(), refres.outliers, res.filter_by_detected);
    EXPECT_EQ(refres.lower_thresholds, res.detected_thresholds);

    // Lowering the nmads to get some actual filtering happening.
    auto res2 = filters.set_nmads(0.5).run(qcres);

    refres = ref.set_nmads(0.5).run(mat->ncol(), qcres.sums);
    compare_vectors(mat->ncol(), refres.outliers, res2.filter_by_sums);
    EXPECT_EQ(refres.lower_thresholds, res2.sums_thresholds);

    refres = ref.run(mat->ncol(), qcres.detected);
    compare_vectors(mat->ncol(), refres.outliers, res2.filter_by_detected);
    EXPECT_EQ(refres.lower_thresholds, res2.detected_thresholds);
}

std::vector<int> keep_i = { 0, 1, 2, 3, 11, 12, 13, 14 };

TEST_F(PerCellQCFiltersTester, OneSubset) {
    std::vector<uint8_t> keep_s(mat->nrow());
    for (auto i : keep_i) { keep_s[i] = 1; }
    auto qcres = qc.set_subsets(std::vector<const uint8_t*>(1, keep_s.data())).run(mat.get());

    scran::PerCellQCFilters filters;
    auto res = filters.set_nmads(1).run(qcres);

    scran::IsOutlier<> ref;
    auto refres = ref.set_nmads(1).set_lower(false).run(mat->ncol(), qcres.subset_proportions[0]);
    compare_vectors(mat->ncol(), refres.outliers, res.filter_by_subset_proportions[0]);
    EXPECT_EQ(refres.upper_thresholds, res.subset_proportions_thresholds[0]);
}

TEST_F(PerCellQCFiltersTester, Blocking) {
    std::vector<uint8_t> keep_s(mat->nrow());
    for (auto i : keep_i) { keep_s[i] = 1; }
    auto qcres = qc.set_subsets(std::vector<const uint8_t*>(1, keep_s.data())).run(mat.get());

    std::vector<int> block(mat->ncol());
    for (size_t i = 0; i < block.size(); ++i) { block[i] = i % 5; }
    scran::PerCellQCFilters filters;
    auto res = filters.set_nmads(1).set_blocks(block.size(), block.data()).run(qcres);

    scran::IsOutlier<> ref;
    auto refres = ref.set_nmads(1).set_lower(true).set_upper(false).set_log(true).set_blocks(block.size(), block.data()).run(mat->ncol(), qcres.sums);
    compare_vectors(mat->ncol(), refres.outliers, res.filter_by_sums);
    EXPECT_EQ(refres.lower_thresholds, res.sums_thresholds);

    refres = ref.run(mat->ncol(), qcres.detected);
    compare_vectors(mat->ncol(), refres.outliers, res.filter_by_detected);
    EXPECT_EQ(refres.lower_thresholds, res.detected_thresholds);

    refres = ref.set_lower(false).set_upper(true).set_log(false).run(mat->ncol(), qcres.subset_proportions[0]);
    compare_vectors(mat->ncol(), refres.outliers, res.filter_by_subset_proportions[0]);
    EXPECT_EQ(refres.upper_thresholds, res.subset_proportions_thresholds[0]);
}

TEST_F(PerCellQCFiltersTester, Overall) {
    std::vector<uint8_t> keep_s(mat->nrow());
    for (auto i : keep_i) { keep_s[i] = 1; }
    auto qcres = qc.set_subsets(std::vector<const uint8_t*>(1, keep_s.data())).run(mat.get());

    scran::PerCellQCFilters filters;
    auto res = filters.run(qcres);

    std::vector<uint8_t> vec(mat->ncol());
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = res.filter_by_sums[i] || res.filter_by_detected[i] || res.filter_by_subset_proportions[0][i];
    }
    compare_vectors(mat->ncol(), res.overall_filter, vec);
}
