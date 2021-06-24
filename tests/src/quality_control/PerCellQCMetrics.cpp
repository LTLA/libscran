#include <gtest/gtest.h>

#include "../data/data.h"

#include "tatami/base/DenseMatrix.hpp"
#include "tatami/base/DelayedSubset.hpp"
#include "tatami/utils/convert_to_dense.hpp"
#include "tatami/utils/convert_to_sparse.hpp"
#include "tatami/stats/sums.hpp"

#include "scran/quality_control/PerCellQCMetrics.hpp"

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

    std::vector<int> to_filter (const std::vector<size_t>& indices) {
        std::vector<int> keep_s(dense_row->nrow());
        for (auto i : indices) { keep_s[i] = 1; }
        return keep_s;        
    }
};

TEST_F(PerCellQCMetricsTester, NoSubset) {
    auto res = qc1.run(dense_row.get());
    EXPECT_EQ(res.sums, tatami::column_sums(dense_row.get()));

    std::vector<int> copy(sparse_matrix.size());
    auto smIt = sparse_matrix.begin();
    for (auto& s : copy) { 
        s = (*smIt > 0); 
        ++smIt;
    }
    auto detected = std::unique_ptr<tatami::typed_matrix<int> >(new tatami::DenseRowMatrix<int>(sparse_nrow, sparse_ncol, copy));
    EXPECT_EQ(res.detected, tatami::column_sums(detected.get()));

    auto res2 = qc2.run(dense_column.get());
    EXPECT_EQ(res.sums, res2.sums);
    EXPECT_EQ(res.detected, res2.detected);

    auto res3 = qc3.run(sparse_row.get());
    EXPECT_EQ(res.sums, res3.sums);
    EXPECT_EQ(res.detected, res3.detected);
    
    auto res4 = qc4.run(sparse_column.get());
    EXPECT_EQ(res.sums, res4.sums);
    EXPECT_EQ(res.detected, res4.detected);
}

TEST_F(PerCellQCMetricsTester, OneSubset) {
    std::vector<size_t> keep_i = { 0, 5, 7, 8, 9, 10, 16, 17 };
    auto keep_s = to_filter(keep_i);
    std::vector<const int*> subs(1, keep_s.data());
    auto res = qc1.set_subsets(subs).run(dense_row.get());

    auto ref = tatami::make_DelayedSubset<0>(dense_row, keep_i);
    auto refprop = tatami::column_sums(ref.get());
    {
        auto sIt = res.sums.begin();
        for (auto& r : refprop) {
            r /= *sIt;
            ++sIt;
        }
    }
    EXPECT_EQ(refprop, res.subset_proportions[0]);

    auto res2 = qc2.set_subsets(subs).run(dense_column.get());
    EXPECT_EQ(refprop, res2.subset_proportions[0]);

    auto res3 = qc3.set_subsets(subs).run(sparse_row.get());
    EXPECT_EQ(refprop, res3.subset_proportions[0]);
    
    auto res4 = qc4.set_subsets(subs).run(sparse_column.get());
    EXPECT_EQ(refprop, res4.subset_proportions[0]);
}

TEST_F(PerCellQCMetricsTester, TwoSubsets) {
    std::vector<size_t> keep_i1 = { 0, 5, 7, 8, 9, 10, 16, 17 };
    std::vector<size_t> keep_i2 = { 1, 8, 2, 6, 11, 5, 19, 17 };
    auto keep_s1 = to_filter(keep_i1), keep_s2 = to_filter(keep_i2);
    std::vector<const int*> subs = { keep_s1.data(), keep_s2.data() };
    auto res = qc1.set_subsets(subs).run(dense_row.get());

    auto ref1 = tatami::make_DelayedSubset<0>(dense_row, keep_i1);
    auto refprop1 = tatami::column_sums(ref1.get());
    {
        auto s1It = res.sums.begin();
        for (auto& r : refprop1) {
            r /= *s1It;
            ++s1It;
        }
    }
    EXPECT_EQ(refprop1, res.subset_proportions[0]);

    auto ref2 = tatami::make_DelayedSubset<0>(dense_row, keep_i2);
    auto refprop2 = tatami::column_sums(ref2.get());
    {
        auto s2It = res.sums.begin();
        for (auto& r : refprop2) {
            r /= *s2It;
            ++s2It;
        }
    }
    EXPECT_EQ(refprop2, res.subset_proportions[1]);

    auto res2 = qc2.set_subsets(subs).run(dense_column.get());
    EXPECT_EQ(refprop1, res2.subset_proportions[0]);
    EXPECT_EQ(refprop2, res2.subset_proportions[1]);

    auto res3 = qc3.set_subsets(subs).run(sparse_row.get());
    EXPECT_EQ(refprop1, res3.subset_proportions[0]);
    EXPECT_EQ(refprop2, res3.subset_proportions[1]);
    
    auto res4 = qc4.set_subsets(subs).run(sparse_column.get());
    EXPECT_EQ(refprop1, res4.subset_proportions[0]);
    EXPECT_EQ(refprop2, res4.subset_proportions[1]);
}

TEST_F(PerCellQCMetricsTester, NASubsets) {
    std::vector<size_t> keep_i = { 0, 5, 7, 8, 9, 10, 16, 17 };
    auto keep_s = to_filter(keep_i);

    std::vector<double> nothing(100);
    auto dense_zero = std::unique_ptr<tatami::numeric_matrix>(new tatami::DenseColumnMatrix<double>(20, 5, std::move(nothing)));

    std::vector<const int*> subs = { keep_s.data() };
    scran::PerCellQCMetrics<int> QC;
    auto res = QC.set_subsets(subs).run(dense_zero.get());

    EXPECT_EQ(res.sums, std::vector<double>(dense_zero->ncol()));
    EXPECT_EQ(res.detected, std::vector<int>(dense_zero->ncol()));
    EXPECT_TRUE(std::isnan(res.subset_proportions[0][0]));
    EXPECT_TRUE(std::isnan(res.subset_proportions[0][dense_zero->ncol()-1]));
}

