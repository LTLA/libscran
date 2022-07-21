#include <gtest/gtest.h>

#include "../data/data.h"

#include "tatami/base/DenseMatrix.hpp"
#include "tatami/base/DelayedSubset.hpp"
#include "tatami/utils/convert_to_dense.hpp"
#include "tatami/utils/convert_to_sparse.hpp"
#include "tatami/stats/sums.hpp"

#include "scran/quality_control/PerCellRnaQcMetrics.hpp"
#include "scran/quality_control/PerCellAdtQcMetrics.hpp"

#include <cmath>

class PerCellRnaQcMetricsTest : public ::testing::TestWithParam<int> {
protected:
    void SetUp() {
        dense_row = std::unique_ptr<tatami::NumericMatrix>(new tatami::DenseRowMatrix<double>(sparse_nrow, sparse_ncol, sparse_matrix));
        dense_column = tatami::convert_to_dense(dense_row.get(), 1);
        sparse_row = tatami::convert_to_sparse(dense_row.get(), 0);
        sparse_column = tatami::convert_to_sparse(dense_row.get(), 1);
    }
protected:
    std::shared_ptr<tatami::NumericMatrix> dense_row, dense_column, sparse_row, sparse_column;

    std::vector<int> to_filter (const std::vector<size_t>& indices) {
        std::vector<int> keep_s(dense_row->nrow());
        for (auto i : indices) { keep_s[i] = 1; }
        return keep_s;        
    }

    template<class Result>
    void compare(const Result& res, const Result& other) {
        EXPECT_EQ(res.sums, other.sums);
        EXPECT_EQ(res.detected, other.detected);
        ASSERT_EQ(res.subset_proportions.size(), other.subset_proportions.size());
        for (size_t i = 0; i < res.subset_proportions.size(); ++i) {
            EXPECT_EQ(res.subset_proportions[i], other.subset_proportions[i]);
        }
    }
};

TEST_P(PerCellRnaQcMetricsTest, NoSubset) {
    scran::PerCellRnaQcMetrics qcfun;
    auto res = qcfun.run(dense_column.get(), {});

    int threads = GetParam();
    qcfun.set_num_threads(threads);

    if (threads == 1) {
        EXPECT_EQ(res.sums, tatami::column_sums(dense_row.get()));

        std::vector<int> copy(sparse_matrix.size());
        auto smIt = sparse_matrix.begin();
        for (auto& s : copy) { 
            s = (*smIt > 0); 
            ++smIt;
        }

        auto detected = std::unique_ptr<tatami::Matrix<int> >(new tatami::DenseRowMatrix<int>(sparse_nrow, sparse_ncol, copy));
        auto refsums = tatami::column_sums(detected.get());
        EXPECT_EQ(res.detected, std::vector<int>(refsums.begin(), refsums.end())); // as column_sums always yeilds a vector of ints.
    } else {
        auto res1 = qcfun.run(dense_column.get(), {});
        compare(res, res1);
    }

    auto res2 = qcfun.run(dense_column.get(), {});
    compare(res, res2);

    auto res3 = qcfun.run(sparse_row.get(), {});
    compare(res, res3);
    
    auto res4 = qcfun.run(sparse_column.get(), {});
    compare(res, res3);
}

TEST_P(PerCellRnaQcMetricsTest, OneSubset) {
    std::vector<size_t> keep_i = { 0, 5, 7, 8, 9, 10, 16, 17 };
    auto keep_s = to_filter(keep_i);
    std::vector<const int*> subs(1, keep_s.data());

    scran::PerCellRnaQcMetrics qcfun;
    auto res = qcfun.run(dense_row.get(), subs);

    int threads = GetParam();
    qcfun.set_num_threads(threads);

    if (threads == 1) {
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
    } else {
        auto res1 = qcfun.run(dense_column.get(), subs);
        compare(res, res1);
    }

    auto res2 = qcfun.run(dense_column.get(), subs);
    compare(res, res2);

    auto res3 = qcfun.run(sparse_row.get(), subs);
    compare(res, res3);
    
    auto res4 = qcfun.run(sparse_column.get(), subs);
    compare(res, res4);
}

TEST_P(PerCellRnaQcMetricsTest, SubsetTotals) {
    std::vector<size_t> keep_i = { 0, 5, 7, 8, 9, 10, 16, 17 };
    auto keep_s = to_filter(keep_i);
    std::vector<const int*> subs(1, keep_s.data());

    scran::PerCellRnaQcMetrics qcfun;
    qcfun.set_subset_totals(true);
    auto res = qcfun.run(dense_row.get(), subs);

    int threads = GetParam();
    qcfun.set_num_threads(threads);

    if (threads == 1) {
        auto ref = tatami::make_DelayedSubset<0>(dense_row, keep_i);
        auto refprop = tatami::column_sums(ref.get());
        EXPECT_EQ(refprop, res.subset_proportions[0]);
    } else {
        auto res1 = qcfun.run(dense_column.get(), subs);
        compare(res, res1);
    }

    auto res2 = qcfun.run(dense_column.get(), subs);
    compare(res, res2);

    auto res3 = qcfun.run(sparse_row.get(), subs);
    compare(res, res3);

    auto res4 = qcfun.run(sparse_column.get(), subs);
    compare(res, res4);

    // Default for the ADT class.
    scran::PerCellAdtQcMetrics qc_adt;
    qc_adt.set_num_threads(threads);
    auto adt_res = qc_adt.run(dense_row.get(), subs);
    EXPECT_EQ(res.subset_proportions[0], adt_res.subset_totals[0]);
}

TEST_P(PerCellRnaQcMetricsTest, TwoSubsets) {
    std::vector<size_t> keep_i1 = { 0, 5, 7, 8, 9, 10, 16, 17 };
    std::vector<size_t> keep_i2 = { 1, 8, 2, 6, 11, 5, 19, 17 };
    auto keep_s1 = to_filter(keep_i1), keep_s2 = to_filter(keep_i2);
    std::vector<const int*> subs = { keep_s1.data(), keep_s2.data() };

    scran::PerCellRnaQcMetrics qcfun;
    auto res = qcfun.run(dense_row.get(), subs);

    int threads = GetParam();
    qcfun.set_num_threads(threads);

    if (threads == 1) {
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
    } else {
        auto res1 = qcfun.run(dense_column.get(), subs);
        compare(res, res1);
    }

    auto res2 = qcfun.run(dense_column.get(), subs);
    compare(res, res2);

    auto res3 = qcfun.run(sparse_row.get(), subs);
    compare(res, res3);
    
    auto res4 = qcfun.run(sparse_column.get(), subs);
    compare(res, res4);
}

TEST_P(PerCellRnaQcMetricsTest, NASubsets) {
    std::vector<size_t> keep_i = { 0, 5, 7, 8, 9, 10, 16, 17 };
    auto keep_s = to_filter(keep_i);

    std::vector<double> nothing(100);
    auto dense_zero = std::unique_ptr<tatami::NumericMatrix>(new tatami::DenseColumnMatrix<double>(20, 5, std::move(nothing)));
    std::vector<const int*> subs = { keep_s.data() };

    scran::PerCellRnaQcMetrics qcfun;
    qcfun.set_num_threads(GetParam());
    auto res = qcfun.run(dense_zero.get(), subs);

    EXPECT_EQ(res.sums, std::vector<double>(dense_zero->ncol()));
    EXPECT_EQ(res.detected, std::vector<int>(dense_zero->ncol()));
    EXPECT_TRUE(std::isnan(res.subset_proportions[0][0]));
    EXPECT_TRUE(std::isnan(res.subset_proportions[0][dense_zero->ncol()-1]));
}

INSTANTIATE_TEST_CASE_P(
    PerCellRnaQcMetrics,
    PerCellRnaQcMetricsTest,
    ::testing::Values(1, 3) // number of threads
);


