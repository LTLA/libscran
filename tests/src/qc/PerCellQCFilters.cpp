#include <gtest/gtest.h>

#include "../data/data.h"

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
}
