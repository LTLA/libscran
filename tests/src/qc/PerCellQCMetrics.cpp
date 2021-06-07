#include <gtest/gtest.h>

#include "../data/data.h"

#include "tatami/base/DenseMatrix.hpp"
#include "tatami/base/DelayedSubset.hpp"
#include "tatami/utils/convert_to_dense.hpp"
#include "tatami/utils/convert_to_sparse.hpp"
#include "tatami/stats/sums.hpp"

#include "scran/qc/PerCellQCMetrics.hpp"

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
    scran::PerCellQCMetrics<int> qc1, qc2, qc3, qc4;
};

TEST_F(PerCellQCMetricsTester, NoSubset) {
    qc1.compute_metrics(dense_row.get());
    EXPECT_EQ(qc1.get_sums(), tatami::column_sums(dense_row.get()));

    std::vector<int> copy(sparse_matrix.size());
    auto smIt = sparse_matrix.begin();
    for (auto& s : copy) { 
        s = (*smIt > 0); 
        ++smIt;
    }
    auto detected = std::unique_ptr<tatami::typed_matrix<int> >(new tatami::DenseRowMatrix<int>(sparse_nrow, sparse_ncol, copy));
    EXPECT_EQ(qc1.get_detected(), tatami::column_sums(detected.get()));

    qc2.compute_metrics(dense_column.get());
    EXPECT_EQ(qc1.get_sums(), qc2.get_sums());
    EXPECT_EQ(qc1.get_detected(), qc2.get_detected());

    qc3.compute_metrics(sparse_row.get());
    EXPECT_EQ(qc1.get_sums(), qc3.get_sums());
    EXPECT_EQ(qc1.get_detected(), qc3.get_detected());
    
    qc4.compute_metrics(sparse_column.get());
    EXPECT_EQ(qc1.get_sums(), qc4.get_sums());
    EXPECT_EQ(qc1.get_detected(), qc4.get_detected());
}

TEST_F(PerCellQCMetricsTester, OneSubset) {
    std::vector<size_t> keep_i = { 0, 5, 7, 8, 9, 10, 16, 17 };
    std::vector<int> keep_s(dense_row->nrow());
    for (auto i : keep_i) { keep_s[i] = 1; }

    std::vector<const int*> subs(1, keep_s.data());
    qc1.set_subsets(subs).compute_metrics(dense_row.get());

    tatami::DelayedSubset<0, double> ref(dense_row, keep_i);
    auto refprop = tatami::column_sums(&ref);
    auto sIt = qc1.get_sums().begin();
    for (auto& r : refprop) {
        r /= *sIt;
        ++sIt;
    }
    EXPECT_EQ(refprop, qc1.get_subset_proportions()[0]);

    qc2.set_subsets(subs).compute_metrics(dense_column.get());
    EXPECT_EQ(refprop, qc2.get_subset_proportions()[0]);

    qc3.set_subsets(subs).compute_metrics(sparse_row.get());
    EXPECT_EQ(refprop, qc3.get_subset_proportions()[0]);
    
    qc4.set_subsets(subs).compute_metrics(sparse_column.get());
    EXPECT_EQ(refprop, qc4.get_subset_proportions()[0]);
}

TEST_F(PerCellQCMetricsTester, TwoSubsets) {
    std::vector<size_t> keep_i1 = { 0, 5, 7, 8, 9, 10, 16, 17 };
    std::vector<size_t> keep_i2 = { 1, 8, 2, 6, 11, 5, 19, 17 };
    std::vector<int> keep_s1(dense_row->nrow()), keep_s2(dense_row->nrow());
    for (auto i : keep_i1) { keep_s1[i] = 1; }
    for (auto i : keep_i2) { keep_s2[i] = 1; }

    std::vector<const int*> subs = { keep_s1.data(), keep_s2.data() };
    qc1.set_subsets(subs).compute_metrics(dense_row.get());

    tatami::DelayedSubset<0, double> ref1(dense_row, keep_i1);
    auto refprop1 = tatami::column_sums(&ref1);
    auto s1It = qc1.get_sums().begin();
    for (auto& r : refprop1) {
        r /= *s1It;
        ++s1It;
    }
    EXPECT_EQ(refprop1, qc1.get_subset_proportions()[0]);

    tatami::DelayedSubset<0, double> ref2(dense_row, keep_i2);
    auto refprop2 = tatami::column_sums(&ref2);
    auto s2It = qc1.get_sums().begin();
    for (auto& r : refprop2) {
        r /= *s2It;
        ++s2It;
    }
    EXPECT_EQ(refprop2, qc1.get_subset_proportions()[1]);

    qc2.set_subsets(subs).compute_metrics(dense_column.get());
    EXPECT_EQ(refprop1, qc2.get_subset_proportions()[0]);
    EXPECT_EQ(refprop2, qc2.get_subset_proportions()[1]);

    qc3.set_subsets(subs).compute_metrics(sparse_row.get());
    EXPECT_EQ(refprop1, qc3.get_subset_proportions()[0]);
    EXPECT_EQ(refprop2, qc3.get_subset_proportions()[1]);
    
    qc4.set_subsets(subs).compute_metrics(sparse_column.get());
    EXPECT_EQ(refprop1, qc4.get_subset_proportions()[0]);
    EXPECT_EQ(refprop2, qc4.get_subset_proportions()[1]);
}

TEST_F(PerCellQCMetricsTester, NASubsets) {
    std::vector<size_t> keep_i = { 0, 5, 7, 8, 9, 10, 16, 17 };
    std::vector<int> keep_s(dense_row->nrow());
    for (auto i : keep_i) { keep_s[i] = 1; }

    std::vector<double> nothing(100);
    auto dense_zero = std::unique_ptr<tatami::numeric_matrix>(new tatami::DenseColumnMatrix<double>(20, 5, std::move(nothing)));

    std::vector<const int*> subs = { keep_s.data() };
    scran::PerCellQCMetrics<int> QC;
    QC.set_subsets(subs).compute_metrics(dense_zero.get());

    EXPECT_EQ(QC.get_sums(), std::vector<double>(dense_zero->ncol()));
    EXPECT_EQ(QC.get_detected(), std::vector<int>(dense_zero->ncol()));
    EXPECT_TRUE(std::isnan(QC.get_subset_proportions()[0][0]));
    EXPECT_TRUE(std::isnan(QC.get_subset_proportions()[0][dense_zero->ncol()-1]));
}

