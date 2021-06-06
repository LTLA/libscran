#include <gtest/gtest.h>

#include "../data/data.h"

#include "tatami/base/DenseMatrix.hpp"
#include "tatami/base/DelayedSubset.hpp"
#include "tatami/utils/convert_to_dense.hpp"
#include "tatami/utils/convert_to_sparse.hpp"
#include "tatami/stats/sums.hpp"

#include "scran/qc/per_cell_qc_metrics.hpp"

#include <cmath>

class PerCellQCMetricsTester : public ::testing::Test {
protected:
    void SetUp() {
        dense_row = std::unique_ptr<tatami::numeric_matrix>(new tatami::DenseRowMatrix<double>(sparse_nrow, sparse_ncol, sparse_matrix));
        dense_column = tatami::convert_to_dense(dense_row.get(), false);
        sparse_row = tatami::convert_to_sparse(dense_row.get(), true);
        sparse_column = tatami::convert_to_sparse(dense_row.get(), false);
    }
protected:
    std::shared_ptr<tatami::numeric_matrix> dense_row, dense_column, sparse_row, sparse_column;
    scran::PerCellQCMetrics<double, int> fdr, fdc, fsr, fsc;
};

TEST_F(PerCellQCMetricsTester, NoSubset) {
    std::vector<const bool*> subs;
    scran::per_cell_qc_metrics(dense_row.get(), subs, fdr);
    EXPECT_EQ(fdr.sums, tatami::column_sums(dense_row.get()));

    std::vector<int> copy(sparse_matrix.size());
    auto smIt = sparse_matrix.begin();
    for (auto& s : copy) { 
        s = (*smIt > 0); 
        ++smIt;
    }
    auto detected = std::unique_ptr<tatami::typed_matrix<int> >(new tatami::DenseRowMatrix<int>(sparse_nrow, sparse_ncol, copy));
    EXPECT_EQ(fdr.detected, tatami::column_sums(detected.get()));

    scran::per_cell_qc_metrics(dense_column.get(), subs, fdc);
    EXPECT_EQ(fdr.sums, fdc.sums);
    EXPECT_EQ(fdr.detected, fdc.detected);

    scran::per_cell_qc_metrics(sparse_row.get(), subs, fsr);
    EXPECT_EQ(fdr.sums, fsr.sums);
    EXPECT_EQ(fdr.detected, fsr.detected);
    
    scran::per_cell_qc_metrics(sparse_column.get(), subs, fsc);
    EXPECT_EQ(fdr.sums, fsc.sums);
    EXPECT_EQ(fdr.detected, fsc.detected);
}

TEST_F(PerCellQCMetricsTester, OneSubset) {
    std::vector<size_t> keep_i = { 0, 5, 7, 8, 9, 10, 16, 17 };
    std::vector<int> keep_s(dense_row->nrow());
    for (auto i : keep_i) { keep_s[i] = 1; }

    std::vector<const int*> subs(1, keep_s.data());
    scran::per_cell_qc_metrics(dense_row.get(), subs, fdr);

    tatami::DelayedSubset<0, double> ref(dense_row, keep_i);
    auto refprop = tatami::column_sums(&ref);
    auto sIt = fdr.sums.begin();
    for (auto& r : refprop) {
        r /= *sIt;
        ++sIt;
    }
    EXPECT_EQ(refprop, fdr.subset_proportions[0]);

    scran::per_cell_qc_metrics(dense_column.get(), subs, fdc);
    EXPECT_EQ(fdr.sums, fdc.sums);
    EXPECT_EQ(fdr.detected, fdc.detected);

    scran::per_cell_qc_metrics(sparse_row.get(), subs, fsr);
    EXPECT_EQ(fdr.sums, fsr.sums);
    EXPECT_EQ(fdr.detected, fsr.detected);
    
    scran::per_cell_qc_metrics(sparse_column.get(), subs, fsc);
    EXPECT_EQ(fdr.sums, fsc.sums);
    EXPECT_EQ(fdr.detected, fsc.detected);
}

TEST_F(PerCellQCMetricsTester, TwoSubsets) {
    std::vector<size_t> keep_i1 = { 0, 5, 7, 8, 9, 10, 16, 17 };
    std::vector<size_t> keep_i2 = { 1, 8, 2, 6, 11, 5, 19, 17 };
    std::vector<int> keep_s1(dense_row->nrow()), keep_s2(dense_row->nrow());
    for (auto i : keep_i1) { keep_s1[i] = 1; }
    for (auto i : keep_i2) { keep_s2[i] = 1; }

    std::vector<const int*> subs = { keep_s1.data(), keep_s2.data() };
    scran::per_cell_qc_metrics(dense_row.get(), subs, fdr);

    tatami::DelayedSubset<0, double> ref1(dense_row, keep_i1);
    auto refprop1 = tatami::column_sums(&ref1);
    auto s1It = fdr.sums.begin();
    for (auto& r : refprop1) {
        r /= *s1It;
        ++s1It;
    }
    EXPECT_EQ(refprop1, fdr.subset_proportions[0]);

    tatami::DelayedSubset<0, double> ref2(dense_row, keep_i2);
    auto refprop2 = tatami::column_sums(&ref2);
    auto s2It = fdr.sums.begin();
    for (auto& r : refprop2) {
        r /= *s2It;
        ++s2It;
    }
    EXPECT_EQ(refprop2, fdr.subset_proportions[1]);

    scran::per_cell_qc_metrics(dense_column.get(), subs, fdc);
    EXPECT_EQ(fdr.sums, fdc.sums);
    EXPECT_EQ(fdr.detected, fdc.detected);

    scran::per_cell_qc_metrics(sparse_row.get(), subs, fsr);
    EXPECT_EQ(fdr.sums, fsr.sums);
    EXPECT_EQ(fdr.detected, fsr.detected);
    
    scran::per_cell_qc_metrics(sparse_column.get(), subs, fsc);
    EXPECT_EQ(fdr.sums, fsc.sums);
    EXPECT_EQ(fdr.detected, fsc.detected);
}

TEST_F(PerCellQCMetricsTester, NASubsets) {
    std::vector<size_t> keep_i = { 0, 5, 7, 8, 9, 10, 16, 17 };
    std::vector<int> keep_s(dense_row->nrow());
    for (auto i : keep_i) { keep_s[i] = 1; }

    std::vector<double> nothing(100);
    auto dense_zero = std::unique_ptr<tatami::numeric_matrix>(new tatami::DenseColumnMatrix<double>(20, 5, std::move(nothing)));

    std::vector<const int*> subs = { keep_s.data() };
    scran::per_cell_qc_metrics(dense_zero.get(), subs, fdr);
    EXPECT_EQ(fdr.sums, std::vector<double>(dense_zero->ncol()));
    EXPECT_EQ(fdr.detected, std::vector<int>(dense_zero->ncol()));
    EXPECT_TRUE(std::isnan(fdr.subset_proportions[0][0]));
    EXPECT_TRUE(std::isnan(fdr.subset_proportions[0][dense_zero->ncol()-1]));
}

