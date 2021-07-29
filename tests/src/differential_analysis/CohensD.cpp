#include <gtest/gtest.h>

#include "tatami/base/Matrix.hpp"
#include "tatami/utils/convert_to_dense.hpp"
#include "tatami/utils/convert_to_sparse.hpp"
#include "scran/differential_analysis/CohensD.hpp"
#include "../data/data.h"

class CohensDTester : public ::testing::TestWithParam<std::tuple<int> > {
protected:
    std::shared_ptr<tatami::NumericMatrix> dense_row, dense_column, sparse_row, sparse_column;

    void SetUp() {
        dense_row = std::unique_ptr<tatami::NumericMatrix>(new tatami::DenseRowMatrix<double>(sparse_nrow, sparse_ncol, sparse_matrix));
        dense_column = tatami::convert_to_dense(dense_row.get(), false);
        sparse_row = tatami::convert_to_sparse(dense_row.get(), true);
        sparse_column = tatami::convert_to_sparse(dense_row.get(), false);
    }
};

TEST_P(CohensDTester, Unblocked) {
    scran::CohensD chd;
    auto ngroups = std::get<0>(GetParam());

    std::vector<int> groupings(dense_row->ncol());
    for (size_t g = 0; g < groupings.size(); ++g) {
        groupings[g] = g % ngroups;
    }

    std::vector<double > output(ngroups * 5 * dense_row->nrow());

    std::vector<std::vector<double*> > ptrs(ngroups, std::vector<double*>(5));
    double* ptr = output.data();
    for (size_t o = 0; o < ngroups; ++o) {
        for (size_t i = 0; i < 5; ++i) {
            ptrs[o][i] = ptr;
            ptr += dense_row->nrow();
        }
    }

    chd.run(dense_row.get(), groupings.data(), std::move(ptrs));
}

INSTANTIATE_TEST_CASE_P(
    CohensD,
    CohensDTester,
    ::testing::Combine(
        ::testing::Values(2, 3, 4) // number of clusters
    )
);

