#include <gtest/gtest.h>
#include "../utils/macros.h"

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
    scran::PerCellAdtQcMetrics qc;

    std::vector<int> keep_i = { 0, 1, 2, 3, 11, 12, 13, 14 };
};

TEST_F(PerCellAdtQcFiltersTester, NoSubset) {
    auto qcres = qc.run(mat.get(), {});

    scran::PerCellAdtQcFilters filters;
    filters.set_nmads(0.5);
    auto res = filters.run(qcres);

    // something got filtered, right?
    EXPECT_TRUE(std::accumulate(res.filter_by_detected.begin(), res.filter_by_detected.end(), 0) > 0); 

    scran::IsOutlier ref;
    ref.set_nmads(0.5).set_lower(true).set_upper(false).set_log(true);
    auto refres = run(ref, qcres.detected);
    EXPECT_EQ(refres.outliers, res.filter_by_detected);
    EXPECT_EQ(refres.thresholds.lower, res.thresholds.detected);
}

TEST_F(PerCellAdtQcFiltersTester, MinDetectedDrop) {
    auto qcres = qc.run(mat.get(), {});
    scran::PerCellAdtQcFilters filters;

    // Get the median...
    filters.set_nmads(0).set_min_detected_drop(0);
    auto res = filters.run(qcres);

    // Applying the minimum drop:
    filters.set_min_detected_drop(0.9);
    auto res2 = filters.run(qcres);
    EXPECT_FLOAT_EQ(res2.thresholds.detected[0], 0.1 * res.thresholds.detected[0]);
}

TEST_F(PerCellAdtQcFiltersTester, OneSubset) {
    std::vector<uint8_t> keep_s(mat->nrow());
    for (auto i : keep_i) { keep_s[i] = 1; }
    auto qcres = qc.run(mat.get(), std::vector<const uint8_t*>(1, keep_s.data()));

    scran::PerCellAdtQcFilters filters;
    auto res = filters.set_nmads(1).run(qcres);
    EXPECT_TRUE(std::accumulate(res.filter_by_subset_totals[0].begin(), res.filter_by_subset_totals[0].end(), 0) > 0); 

    scran::IsOutlier ref;
    ref.set_nmads(1).set_lower(false).set_log(true);
    auto refres = run(ref, qcres.subset_totals[0]);
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
    ref.set_nmads(1).set_lower(true).set_upper(false).set_log(true);
    auto refres = run_blocked(ref, qcres.detected, block.data());
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
