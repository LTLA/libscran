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
    qc.run(mat.get());

    scran::PerCellQCFilters filters;
    filters.run(mat->ncol(), qc);

    scran::IsOutlier<> ref;
    ref.set_lower(true).set_upper(false).set_log(true).run(mat->ncol(), qc.get_sums());
    compare_vectors(mat->ncol(), ref.get_outliers(), filters.get_filter_by_sums());
    EXPECT_EQ(ref.get_lower_thresholds(), filters.get_sums_thresholds());

    ref.run(mat->ncol(), qc.get_detected());
    compare_vectors(mat->ncol(), ref.get_outliers(), filters.get_filter_by_detected());
    EXPECT_EQ(ref.get_lower_thresholds(), filters.get_detected_thresholds());

    // Lowering the nmads to get some actual filtering happening.
    filters.set_nmads(0.5).run(mat->ncol(), qc);

    ref.set_nmads(0.5).run(mat->ncol(), qc.get_sums());
    compare_vectors(mat->ncol(), ref.get_outliers(), filters.get_filter_by_sums());
    EXPECT_EQ(ref.get_lower_thresholds(), filters.get_sums_thresholds());

    ref.run(mat->ncol(), qc.get_detected());
    compare_vectors(mat->ncol(), ref.get_outliers(), filters.get_filter_by_detected());
    EXPECT_EQ(ref.get_lower_thresholds(), filters.get_detected_thresholds());
}

std::vector<int> keep_i = { 0, 1, 2, 3, 11, 12, 13, 14 };

TEST_F(PerCellQCFiltersTester, OneSubset) {
    std::vector<uint8_t> keep_s(mat->nrow());
    for (auto i : keep_i) { keep_s[i] = 1; }
    qc.set_subsets(std::vector<const uint8_t*>(1, keep_s.data())).run(mat.get());

    scran::PerCellQCFilters filters;
    filters.set_nmads(1).run(mat->ncol(), qc);

    scran::IsOutlier<> ref;
    ref.set_nmads(1).set_lower(false).run(mat->ncol(), qc.get_subset_proportions()[0]);
    compare_vectors(mat->ncol(), ref.get_outliers(), filters.get_filter_by_subset_proportions()[0]);
    EXPECT_EQ(ref.get_upper_thresholds(), filters.get_subset_proportions_thresholds()[0]);
}

TEST_F(PerCellQCFiltersTester, Blocking) {
    std::vector<uint8_t> keep_s(mat->nrow());
    for (auto i : keep_i) { keep_s[i] = 1; }
    qc.set_subsets(std::vector<const uint8_t*>(1, keep_s.data())).run(mat.get());

    std::vector<int> block(mat->ncol());
    for (size_t i = 0; i < block.size(); ++i) { block[i] = i % 5; }
    scran::PerCellQCFilters filters;
    filters.set_nmads(1).set_blocks(block.size(), block.data()).run(mat->ncol(), qc);

    scran::IsOutlier<> ref;
    ref.set_nmads(1).set_lower(true).set_upper(false).set_log(true).set_blocks(block.size(), block.data()).run(mat->ncol(), qc.get_sums());
    compare_vectors(mat->ncol(), ref.get_outliers(), filters.get_filter_by_sums());
    EXPECT_EQ(ref.get_lower_thresholds(), filters.get_sums_thresholds());

    ref.run(mat->ncol(), qc.get_detected());
    compare_vectors(mat->ncol(), ref.get_outliers(), filters.get_filter_by_detected());
    EXPECT_EQ(ref.get_lower_thresholds(), filters.get_detected_thresholds());

    ref.set_lower(false).set_upper(true).set_log(false).run(mat->ncol(), qc.get_subset_proportions()[0]);
    compare_vectors(mat->ncol(), ref.get_outliers(), filters.get_filter_by_subset_proportions()[0]);
    EXPECT_EQ(ref.get_upper_thresholds(), filters.get_subset_proportions_thresholds()[0]);
}
