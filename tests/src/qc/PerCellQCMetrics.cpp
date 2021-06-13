#include <gtest/gtest.h>

#include "../data/data.h"
#include "../utils/compare_vectors.h"

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

    std::vector<int> to_filter (const std::vector<size_t>& indices) {
        std::vector<int> keep_s(dense_row->nrow());
        for (auto i : indices) { keep_s[i] = 1; }
        return keep_s;        
    }
};

TEST_F(PerCellQCMetricsTester, NoSubset) {
    auto res = qc1.run(dense_row.get());
    compare_vectors(dense_row->ncol(), res.sums, tatami::column_sums(dense_row.get()));

    std::vector<int> copy(sparse_matrix.size());
    auto smIt = sparse_matrix.begin();
    for (auto& s : copy) { 
        s = (*smIt > 0); 
        ++smIt;
    }
    auto detected = std::unique_ptr<tatami::typed_matrix<int> >(new tatami::DenseRowMatrix<int>(sparse_nrow, sparse_ncol, copy));
    compare_vectors(dense_row->ncol(), res.detected, tatami::column_sums(detected.get()));

    auto res2 = qc2.run(dense_column.get());
    compare_vectors(dense_row->ncol(), res.sums, res2.sums);
    compare_vectors(dense_row->ncol(), res.detected, res2.detected);

    auto res3 = qc3.run(sparse_row.get());
    compare_vectors(dense_row->ncol(), res.sums, res3.sums);
    compare_vectors(dense_row->ncol(), res.detected, res3.detected);
    
    auto res4 = qc4.run(sparse_column.get());
    compare_vectors(dense_row->ncol(), res.sums, res4.sums);
    compare_vectors(dense_row->ncol(), res.detected, res4.detected);
}

TEST_F(PerCellQCMetricsTester, OneSubset) {
    std::vector<size_t> keep_i = { 0, 5, 7, 8, 9, 10, 16, 17 };
    auto keep_s = to_filter(keep_i);
    std::vector<const int*> subs(1, keep_s.data());
    auto res = qc1.set_subsets(subs).run(dense_row.get());

    auto ref = tatami::make_DelayedSubset<0>(dense_row, keep_i);
    auto refprop = tatami::column_sums(ref.get());
    {
        auto sIt = res.sums;
        for (auto& r : refprop) {
            r /= *sIt;
            ++sIt;
        }
    }
    compare_vectors(refprop, dense_row->ncol(), res.subset_proportions[0]);

    auto res2 = qc2.set_subsets(subs).run(dense_column.get());
    compare_vectors(refprop, dense_row->ncol(), res2.subset_proportions[0]);

    auto res3 = qc3.set_subsets(subs).run(sparse_row.get());
    compare_vectors(refprop, dense_row->ncol(), res3.subset_proportions[0]);
    
    auto res4 = qc4.set_subsets(subs).run(sparse_column.get());
    compare_vectors(refprop, dense_row->ncol(), res4.subset_proportions[0]);
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
        auto s1It = res.sums;
        for (auto& r : refprop1) {
            r /= *s1It;
            ++s1It;
        }
    }
    compare_vectors(refprop1, dense_row->ncol(), res.subset_proportions[0]);

    auto ref2 = tatami::make_DelayedSubset<0>(dense_row, keep_i2);
    auto refprop2 = tatami::column_sums(ref2.get());
    {
        auto s2It = res.sums;
        for (auto& r : refprop2) {
            r /= *s2It;
            ++s2It;
        }
    }
    compare_vectors(refprop2, dense_row->ncol(), res.subset_proportions[1]);

    auto res2 = qc2.set_subsets(subs).run(dense_column.get());
    compare_vectors(refprop1, dense_row->ncol(), res2.subset_proportions[0]);
    compare_vectors(refprop2, dense_row->ncol(), res2.subset_proportions[1]);

    auto res3 = qc3.set_subsets(subs).run(sparse_row.get());
    compare_vectors(refprop1, dense_row->ncol(), res3.subset_proportions[0]);
    compare_vectors(refprop2, dense_row->ncol(), res3.subset_proportions[1]);
    
    auto res4 = qc4.set_subsets(subs).run(sparse_column.get());
    compare_vectors(refprop1, dense_row->ncol(), res4.subset_proportions[0]);
    compare_vectors(refprop2, dense_row->ncol(), res4.subset_proportions[1]);
}

TEST_F(PerCellQCMetricsTester, NASubsets) {
    std::vector<size_t> keep_i = { 0, 5, 7, 8, 9, 10, 16, 17 };
    auto keep_s = to_filter(keep_i);

    std::vector<double> nothing(100);
    auto dense_zero = std::unique_ptr<tatami::numeric_matrix>(new tatami::DenseColumnMatrix<double>(20, 5, std::move(nothing)));

    std::vector<const int*> subs = { keep_s.data() };
    scran::PerCellQCMetrics<int> QC;
    auto res = QC.set_subsets(subs).run(dense_zero.get());

    compare_vectors(dense_zero->ncol(), res.sums, std::vector<double>(dense_zero->ncol()));
    compare_vectors(dense_zero->ncol(), res.detected, std::vector<int>(dense_zero->ncol()));
    EXPECT_TRUE(std::isnan(res.subset_proportions[0][0]));
    EXPECT_TRUE(std::isnan(res.subset_proportions[0][dense_zero->ncol()-1]));
}

